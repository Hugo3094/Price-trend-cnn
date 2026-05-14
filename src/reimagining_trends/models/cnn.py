"""
cnn.py
------
CNN 2D pour la prédiction de rendements à partir d'images OHLC.
Architecture fidèle au papier Jiang, Kelly & Xiu (2023).

Dimensions des images :
  - 5  jours : 32×15
  - 20 jours : 64×60
  - 60 jours : 96×180

Architecture par fenêtre :
  - 5j  : 2 blocs CNN (64, 128  filtres)
  - 20j : 3 blocs CNN (64, 128, 256 filtres)
  - 60j : 4 blocs CNN (64, 128, 256, 512 filtres)

Chaque bloc = Conv(5×3) → BatchNorm → LeakyReLU → MaxPool(2×1)

Référence : Jiang, Kelly & Xiu (2023), Section II & Appendix
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Bloc de base CNN
# ---------------------------------------------------------------------------
class CNNBlock(nn.Module):
    """
    Bloc de base : Conv2D → BatchNorm → LeakyReLU → MaxPool.
    Filtre 5×3 (hauteur × largeur), MaxPool 2×1.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  tuple[int, int] = (5, 3),
        pool_size:    tuple[int, int] = (2, 1),
        dilation:     tuple[int, int] = (1, 1),
        leaky_slope:  float = 0.01,
    ) -> None:
        super().__init__()

        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size = kernel_size,
            padding     = padding,
            dilation    = dilation,
            bias        = False,
        )
        self.bn      = nn.BatchNorm2d(out_channels)
        self.act     = nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)
        self.pool    = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.bn(self.conv(x))))


# ---------------------------------------------------------------------------
# CNN principal
# ---------------------------------------------------------------------------
class PriceCNN(nn.Module):
    """
    CNN 2D pour images OHLC.

    L'architecture est automatiquement sélectionnée selon la taille de la fenêtre
    (2, 3 ou 4 blocs), conformément au papier.
    """

    # Configurations des blocs par fenêtre
    CONFIGS = {
        5:  {"n_blocks": 2, "filter_base": 64, "dilation": (2, 1)},
        20: {"n_blocks": 3, "filter_base": 64, "dilation": (2, 1)},
        60: {"n_blocks": 4, "filter_base": 64, "dilation": (3, 1)},
    }

    def __init__(
        self,
        window:       int = 20,
        image_height: int = 60,
        image_width:  int = 64,
        in_channels:  int = 1,
        dropout:      float = 0.5,
        n_classes:    int = 2,
    ) -> None:
        """
        Parameters
        ----------
        window       : 5, 20 ou 60
        image_height : hauteur de l'image en pixels
        image_width  : largeur de l'image en pixels
        in_channels  : 1 (niveaux de gris)
        dropout      : dropout sur la couche FC
        n_classes    : 2 (up / down)
        """
        super().__init__()

        assert window in self.CONFIGS, f"window doit être dans {list(self.CONFIGS.keys())}"
        cfg = self.CONFIGS[window]

        # Construction des blocs CNN
        blocks       = []
        in_ch        = in_channels
        out_ch       = cfg["filter_base"]
        dilation_h   = cfg["dilation"][0]

        for i in range(cfg["n_blocks"]):
            # Dilation seulement sur le premier bloc (images creuses)
            dil = (dilation_h, 1) if i == 0 else (1, 1)
            blocks.append(CNNBlock(in_ch, out_ch, dilation=dil))
            in_ch  = out_ch
            out_ch = out_ch * 2          # doublement des filtres à chaque bloc

        self.cnn_blocks = nn.Sequential(*blocks)

        # Calcul de la dimension de sortie des blocs CNN
        fc_in = self._get_fc_input_dim(image_height, image_width, in_channels, cfg["n_blocks"])

        # Couche fully connected + softmax
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(fc_in, n_classes)

    def _get_fc_input_dim(
        self,
        h: int, w: int,
        in_channels: int,
        n_blocks: int,
    ) -> int:
        """Calcule la dimension d'entrée de la FC par passage d'un tenseur dummy."""
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, h, w)
            out   = self.cnn_blocks(dummy)
            return int(out.numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, 1, H, W)   — images OHLC

        Returns
        -------
        logits : (batch, 2)
        """
        feat = self.cnn_blocks(x)
        feat = self.flatten(feat)
        feat = self.dropout(feat)
        return self.fc(feat)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extrait les feature maps avant la FC (utile pour Grad-CAM)."""
        return self.cnn_blocks(x)


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
class GradCAM:
    """
    Implémentation de Grad-CAM pour visualiser les zones actives
    dans les images OHLC qui influencent la prédiction du CNN.

    Référence : Selvaraju et al. (2017) — Grad-CAM
    """

    def __init__(self, model: PriceCNN, target_layer: Optional[nn.Module] = None) -> None:
        self.model        = model
        self.target_layer = target_layer or list(model.cnn_blocks.children())[-1].conv
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, x: torch.Tensor, class_idx: Optional[int] = None) -> torch.Tensor:
        """
        Génère la carte de chaleur Grad-CAM.

        Parameters
        ----------
        x         : (1, 1, H, W)
        class_idx : indice de la classe cible (None = classe prédite)

        Returns
        -------
        cam : (H, W)  — carte de chaleur normalisée [0, 1]
        """
        self.model.eval()
        x = x.requires_grad_(True)

        logits = self.model(x)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Poids : moyenne des gradients sur H et W
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)   # (1, C, 1, 1)

        # Combinaison linéaire des feature maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        cam = F.relu(cam)

        # Upsample vers la taille de l'image originale
        cam = F.interpolate(cam, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
        cam = cam.squeeze()

        # Normalisation
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------
def build_cnn(
    window:   int = 20,
    dropout:  float = 0.5,
) -> PriceCNN:
    """
    Construit le CNN avec les bonnes dimensions d'image selon la fenêtre.
    """
    from imaging.ohlc_chart import IMAGE_SPECS
    specs = IMAGE_SPECS[window]
    return PriceCNN(
        window       = window,
        image_height = specs["height"],
        image_width  = specs["width"],
        in_channels  = 1,
        dropout      = dropout,
    )


# ---------------------------------------------------------------------------
# Test rapide
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    for window in [5, 20, 60]:
        from imaging.ohlc_chart import IMAGE_SPECS
        specs = IMAGE_SPECS[window]
        H, W  = specs["height"], specs["width"]

        model = build_cnn(window=window)
        x     = torch.randn(4, 1, H, W)
        out   = model(x)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Window={window:2d}j | Image {H}×{W} | Output {out.shape} | Params {n_params:>10,}")

    # Test Grad-CAM
    print("\n=== Test Grad-CAM ===")
    model  = build_cnn(window=20)
    gradcam = GradCAM(model)
    x      = torch.randn(1, 1, 60, 64)
    cam    = gradcam.generate(x)
    print(f"CAM shape : {cam.shape}, min={cam.min():.3f}, max={cam.max():.3f}")

"""
mlp.py
------
Multi-Layer Perceptron sur features tabulaires (représentation "plate").
Sert de baseline simple dans la comparaison des architectures.

Entrée  : (batch, window * n_features)  — séquence aplatie
Sortie  : (batch, 2)                     — logits up/down
"""

import torch
import torch.nn as nn
from typing import Optional


class MLP(nn.Module):
    """
    MLP avec BatchNorm et Dropout entre chaque couche.

    Architecture :
        Input → [Linear → BN → ReLU → Dropout] × n_layers → Linear(2)
    """

    def __init__(
        self,
        input_dim:   int,
        hidden_dims: list[int] = [256, 128, 64],
        dropout:     float = 0.3,
        n_classes:   int = 2,
    ) -> None:
        """
        Parameters
        ----------
        input_dim   : window * n_features (ex. 20 * 6 = 120)
        hidden_dims : liste des dimensions des couches cachées
        dropout     : taux de dropout
        n_classes   : 2 (classification binaire up/down)
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, window, n_features) ou (batch, window * n_features)
        """
        if x.dim() == 3:
            x = x.view(x.size(0), -1)   # aplatissement
        return self.net(x)


def build_mlp(
    window:      int,
    n_features:  int,
    hidden_dims: list[int] = [256, 128, 64],
    dropout:     float = 0.3,
) -> MLP:
    """Factory function."""
    input_dim = window * n_features
    return MLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)


# ---------------------------------------------------------------------------
# Test rapide
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = build_mlp(window=20, n_features=6)
    print(model)
    x = torch.randn(32, 20, 6)      # batch de 32
    out = model(x)
    print(f"Output shape : {out.shape}")   # (32, 2)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres   : {n_params:,}")

"""
Representation Learning Module

Provides unified interface for learning latent representations (embeddings)
from tabular features using methods like PCA, UMAP, and optional Autoencoders.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RepresentationConfig:
    method: str = "umap"  # 'umap', 'pca', or 'autoencoder'
    n_components: int = 16
    random_state: int = 42
    # UMAP-specific
    umap_n_neighbors: Optional[int] = None
    umap_metric: str = "euclidean"
    umap_min_dist: float = 0.1
    # Autoencoder-specific
    ae_hidden_dims: Optional[list[int]] = None
    ae_dropout: float = 0.0
    ae_lr: float = 1e-3
    ae_batch_size: int = 128
    ae_epochs: int = 50


class RepresentationLearner:
    """
    Learn latent representations from tabular data.

    Usage:
        rep = RepresentationLearner(method='umap', n_components=16)
        Z = rep.fit_transform(X_train)
        Z_test = rep.transform(X_test)
    """

    def __init__(self, config: RepresentationConfig):
        self.config = config
        self.model: Optional[Any] = None
        self._fitted = False
        self._method = config.method.lower()

    def _init_model(self, n_features: int) -> Any:
        if self._method == "pca":
            from sklearn.decomposition import PCA

            logger.info(
                f"Initializing PCA representation (n_components={self.config.n_components})"
            )
            return PCA(n_components=self.config.n_components, random_state=self.config.random_state)

        elif self._method == "umap":
            try:
                import umap  # type: ignore

                logger.info(
                    f"Initializing UMAP representation (n_components={self.config.n_components})"
                )
                n_neighbors = self.config.umap_n_neighbors
                if not n_neighbors:
                    n_neighbors = min(15, max(5, int(np.sqrt(len(self._train_index) or 100))))
                return umap.UMAP(
                    n_components=self.config.n_components,
                    random_state=self.config.random_state,
                    n_neighbors=n_neighbors,
                    metric=self.config.umap_metric,
                    min_dist=self.config.umap_min_dist,
                    verbose=False,
                )
            except Exception as e:
                logger.warning(
                    f"UMAP not available or failed to import ({e}); falling back to PCA."
                )
                self._method = "pca"
                return self._init_model(n_features)

        elif self._method == "autoencoder":
            # Try PyTorch first; if not available, fallback
            try:
                import torch
                import torch.nn as nn
                from torch.utils.data import DataLoader, TensorDataset

                input_dim = n_features
                hidden_dims = self.config.ae_hidden_dims or [64, 32]
                latent_dim = self.config.n_components

                class AE(nn.Module):
                    def __init__(self):
                        super().__init__()
                        layers = []
                        last = input_dim
                        for h in hidden_dims:
                            layers += [nn.Linear(last, h), nn.ReLU()]
                            if self.config.ae_dropout > 0:
                                layers += [nn.Dropout(self.config.ae_dropout)]
                            last = h
                        self.encoder = nn.Sequential(*layers)
                        self.latent = nn.Linear(last, latent_dim)

                        dec_layers = []
                        last = latent_dim
                        for h in reversed(hidden_dims):
                            dec_layers += [nn.Linear(last, h), nn.ReLU()]
                            last = h
                        dec_layers += [nn.Linear(last, input_dim)]
                        self.decoder = nn.Sequential(*dec_layers)

                    def forward(self, x):
                        h = self.encoder(x)
                        z = self.latent(h)
                        x_hat = self.decoder(z)
                        return x_hat, z

                # Package training objects into a dict so we can keep a simple API
                model = {
                    "framework": "torch",
                    "net": AE(),
                    "lr": self.config.ae_lr,
                    "batch_size": self.config.ae_batch_size,
                    "epochs": self.config.ae_epochs,
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                    "TensorDataset": TensorDataset,
                    "Tensor": torch.tensor,
                    "DataLoader": DataLoader,
                    "nn": nn,
                }
                logger.info(
                    f"Initializing Autoencoder (latent={latent_dim}, hidden={hidden_dims}, device={model['device']})"
                )
                return model
            except Exception as e:
                logger.warning(
                    f"Autoencoder dependencies not available or failed to init ({e}); falling back to PCA."
                )
                self._method = "pca"
                return self._init_model(n_features)
        else:
            raise ValueError(f"Unknown representation method: {self._method}")

    def fit(self, X: pd.DataFrame) -> "RepresentationLearner":
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise ValueError("X must be a pandas DataFrame or numpy array")
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        n_features = X_np.shape[1]
        self._train_index = getattr(X, "index", None)

        self.model = self._init_model(n_features)

        if isinstance(self.model, dict) and self.model.get("framework") == "torch":
            # Train autoencoder
            torch = __import__("torch")
            nn = self.model["nn"]
            Tensor = self.model["Tensor"]
            DataLoader = self.model["DataLoader"]
            TensorDataset = self.model["TensorDataset"]

            device = self.model["device"]
            net = self.model["net"].to(device)
            opt = torch.optim.Adam(net.parameters(), lr=self.model["lr"])
            loss_fn = nn.MSELoss()

            dataset = TensorDataset(Tensor(X_np).float())
            loader = DataLoader(dataset, batch_size=self.model["batch_size"], shuffle=True)

            net.train()
            for epoch in range(self.model["epochs"]):
                epoch_loss = 0.0
                for (batch,) in loader:
                    batch = batch.to(device)
                    opt.zero_grad()
                    x_hat, _ = net(batch)
                    loss = loss_fn(x_hat, batch)
                    loss.backward()
                    opt.step()
                    epoch_loss += loss.item() * batch.size(0)
                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"AE training epoch {epoch+1}/{self.model['epochs']} - loss={epoch_loss / len(dataset):.6f}"
                    )
        else:
            # PCA/UMAP
            self.model.fit(X_np)

        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("RepresentationLearner must be fitted before transform().")

        X_np = X.values if isinstance(X, pd.DataFrame) else X

        if isinstance(self.model, dict) and self.model.get("framework") == "torch":
            torch = __import__("torch")
            device = self.model["device"]
            net = self.model["net"].to(device)
            net.eval()
            with torch.no_grad():
                X_tensor = self.model["Tensor"](X_np).float().to(device)
                # Forward pass to get latent
                # Re-implement forward pieces to avoid decoding
                h = net.encoder(X_tensor)
                z = net.latent(h)
                Z = z.cpu().numpy()
        else:
            # PCA/UMAP
            if hasattr(self.model, "transform"):
                Z = self.model.transform(X_np)
            else:
                # Some UMAP versions use fit_transform only; ensure transform is available
                Z = self.model.fit_transform(X_np)

        # Return as DataFrame with descriptive names
        cols = [f"rep_{self._method}_z{i+1}" for i in range(Z.shape[1])]
        return pd.DataFrame(Z, columns=cols, index=getattr(X, "index", None))

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


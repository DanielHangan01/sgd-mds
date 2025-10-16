import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

from sgd_mds.estimator import SGDMDS


X = np.load("datasets/seismic/X.npy")
y = np.load("datasets/seismic/y.npy")

n = min(len(X), len(y))
X, y = X[:n], y[:n]
print(f"Loaded seismic dataset: X={X.shape}, y={y.shape}")

print("Computing pairwise distances")
D = pairwise_distances(X, metric="euclidean").astype(np.float32)

print("Running SGD-MDS...")
model = SGDMDS(
    n_components=2,
    max_iter=2000,
    batch_size=50_000,
    lr_init=0.5,
    random_state=42,
    device="auto",
)
X_emb: np.ndarray = model.fit_transform(D)  # type: ignore

print(f"Finished after {model.n_iter_} iterations.")
print(f"Final Stress: {model.stress_:.6f}")

plt.figure(figsize=(6, 5))
plt.scatter(X_emb[:, 0], X_emb[:, 1], c=y, cmap="coolwarm", s=25, alpha=0.9)
plt.title(f"Seismic dataset — SGD-MDS (stress={model.stress_:.4f})")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.tight_layout()
plt.show()

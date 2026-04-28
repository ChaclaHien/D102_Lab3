"""
svm_numpy.py
============
Soft-Margin SVM tự cài đặt bằng NumPy với thuật toán Mini-batch SGD.

Bài toán tối ưu (primal):
    min  ½‖w‖² + C · (1/N) Σ max(0, 1 − yᵢ(wᵀxᵢ + b))

Subgradient trên mini-batch B:
    ∂/∂w = w − (C/|B|) Σ_{i: margin<1} yᵢ xᵢ
    ∂/∂b =   − (C/|B|) Σ_{i: margin<1} yᵢ
"""

import numpy as np


class SoftMarginSVM:
    """
    Soft-Margin SVM huấn luyện bằng Mini-batch SGD.

    Parameters
    ----------
    C          : float – hệ số penalty (regularisation strength)
    lr         : float – learning rate khởi đầu
    n_epochs   : int   – số epoch huấn luyện
    batch_size : int   – kích thước mini-batch
    lr_decay   : float – hệ số giảm learning rate sau mỗi epoch
    """

    def __init__(
        self,
        C: float = 1.0,
        lr: float = 0.01,
        n_epochs: int = 50,
        batch_size: int = 64,
        lr_decay: float = 0.95,
    ):
        self.C          = C
        self.lr0        = lr
        self.n_epochs   = n_epochs
        self.batch_size = batch_size
        self.lr_decay   = lr_decay

        self.w: np.ndarray | None = None
        self.b: float             = 0.0
        self.train_losses: list   = []
        self.val_losses: list     = []

    # ── Hinge loss ────────────────────────────────────────────────────────────
    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        margin = y * (X @ self.w + self.b)
        hinge  = np.maximum(0.0, 1.0 - margin).mean()
        return float(0.5 * self.w @ self.w + self.C * hinge)

    # ── Một bước SGD ──────────────────────────────────────────────────────────
    def _step(self, X_b: np.ndarray, y_b: np.ndarray, lr: float) -> None:
        B      = len(y_b)
        margin = y_b * (X_b @ self.w + self.b)
        mask   = margin < 1   # vị trí hinge đang active

        dw = self.w.copy()
        db = 0.0
        if mask.any():
            dw -= (self.C / B) * (y_b[mask, None] * X_b[mask]).sum(0)
            db  = -(self.C / B) * y_b[mask].sum()

        self.w -= lr * dw
        self.b -= lr * db

    # ── Huấn luyện ────────────────────────────────────────────────────────────
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        verbose: bool = True,
    ) -> "SoftMarginSVM":
        """
        Huấn luyện mô hình.

        Parameters
        ----------
        X, y         – tập huấn luyện
        X_val, y_val – tập validation (tuỳ chọn, để theo dõi val loss)
        verbose      – in thông tin mỗi 10 epoch nếu True
        """
        N, D   = X.shape
        self.w = np.zeros(D)
        self.b = 0.0
        lr     = self.lr0
        yf     = y.astype(np.float64)
        yvf    = y_val.astype(np.float64) if y_val is not None else None

        for ep in range(1, self.n_epochs + 1):
            perm = np.random.permutation(N)
            for s in range(0, N, self.batch_size):
                idx = perm[s : s + self.batch_size]
                self._step(X[idx], yf[idx], lr)

            lr *= self.lr_decay
            self.train_losses.append(self._loss(X, yf))

            if X_val is not None:
                self.val_losses.append(self._loss(X_val, yvf))

            if verbose and (ep % 10 == 0 or ep == 1):
                vs = (
                    f"  val={self.val_losses[-1]:.4f}"
                    if X_val is not None
                    else ""
                )
                print(
                    f"  epoch {ep:>3}/{self.n_epochs} "
                    f"train={self.train_losses[-1]:.4f}{vs}  lr={lr:.5f}"
                )

        print("Training complete ✓")
        return self

    # ── Dự đoán ───────────────────────────────────────────────────────────────
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Trả về giá trị khoảng cách đến siêu phẳng phân loại."""
        return X @ self.w + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Trả về nhãn dự đoán: +1 (PNEUMONIA) hoặc -1 (NORMAL)."""
        return np.sign(self.decision_function(X)).astype(np.int32)

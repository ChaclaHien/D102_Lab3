"""
svm_sklearn.py
==============
Wrapper cho sklearn.svm.SVC, dùng trong Assignment 2
để so sánh với SoftMarginSVM tự cài đặt.
"""

import time
import numpy as np
from sklearn.svm import SVC


def build_sklearn_svm(C: float = 1.0, kernel: str = "rbf", random_state: int = 42) -> SVC:
    """
    Tạo một instance SVC với các tham số cho trước.

    Parameters
    ----------
    C            : float – hệ số penalty
    kernel       : str   – loại kernel ("rbf", "linear", …)
    random_state : int   – seed

    Returns
    -------
    SVC instance chưa được fit
    """
    return SVC(kernel=kernel, C=C, gamma="scale", random_state=random_state)


def train_sklearn_svm(
    model: SVC,
    X_train: np.ndarray,
    y_train: np.ndarray,
    verbose: bool = True,
) -> tuple[SVC, float]:
    """
    Huấn luyện sklearn SVM và đo thời gian.

    Parameters
    ----------
    model   : SVC instance
    X_train : np.ndarray – dữ liệu huấn luyện (đã qua PCA)
    y_train : np.ndarray – nhãn huấn luyện
    verbose : bool       – in thông tin nếu True

    Returns
    -------
    model       : SVC đã được fit
    elapsed_sec : float – thời gian huấn luyện (giây)
    """
    if verbose:
        print("\n[sklearn SVM] Training …")

    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    if verbose:
        print(f"  Time: {elapsed:.1f}s")
        print("sklearn SVM training complete ✓")

    return model, elapsed

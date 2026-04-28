"""
preprocessing.py
================
Tiền xử lý dữ liệu ảnh: chuẩn hoá (StandardScaler) và giảm chiều (PCA).
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def fit_transform_pipeline(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    n_components: int = 150,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler, PCA]:
    """
    Áp dụng StandardScaler rồi PCA.

    Parameters
    ----------
    X_train, X_val, X_test : np.ndarray – dữ liệu thô (N, D)
    n_components            : int        – số chiều PCA (mặc định 150)
    random_state            : int        – seed cho PCA

    Returns
    -------
    X_train_pca, X_val_pca, X_test_pca : np.ndarray – dữ liệu đã biến đổi
    scaler                              : StandardScaler đã fit
    pca                                 : PCA đã fit
    """
    print("\nStandardising …")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

    print(f"PCA → {n_components} components …")
    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_sc)
    X_val_pca   = pca.transform(X_val_sc)
    X_test_pca  = pca.transform(X_test_sc)

    var_exp = np.cumsum(pca.explained_variance_ratio_)[-1] * 100
    print(f"  Variance explained: {var_exp:.1f}%  |  shape: {X_train_pca.shape}")

    return X_train_pca, X_val_pca, X_test_pca, scaler, pca


def fit_transform_combined(
    X_combined: np.ndarray,
    X_test: np.ndarray,
    n_train: int,
    n_components: int = 150,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler, PCA]:
    """
    Áp dụng StandardScaler + PCA khi train và val được gộp chung (Assignment 2).

    Parameters
    ----------
    X_combined  : np.ndarray – dữ liệu train+val gộp lại (N_tr + N_val, D)
    X_test      : np.ndarray – dữ liệu test
    n_train     : int        – số mẫu thuộc tập train (để tách lại sau PCA)
    n_components: int        – số chiều PCA
    random_state: int        – seed

    Returns
    -------
    X_train_pca, X_val_pca, X_test_pca : np.ndarray
    scaler                              : StandardScaler đã fit
    pca                                 : PCA đã fit
    """
    print("\nStandardising + PCA …")
    scaler = StandardScaler()
    X_combined_sc = scaler.fit_transform(X_combined)
    X_test_sc     = scaler.transform(X_test)

    pca = PCA(n_components=n_components, random_state=random_state)
    X_combined_pca = pca.fit_transform(X_combined_sc)
    X_test_pca     = pca.transform(X_test_sc)

    X_train_pca = X_combined_pca[:n_train]
    X_val_pca   = X_combined_pca[n_train:]

    var_exp = np.cumsum(pca.explained_variance_ratio_)[-1] * 100
    print(f"  {n_components} components → {var_exp:.1f}% variance  |  shape: {X_combined_pca.shape}")

    return X_train_pca, X_val_pca, X_test_pca, scaler, pca

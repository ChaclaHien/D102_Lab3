"""
main_as2.py
===========
Assignment 2 – So sánh SVM tự cài đặt (NumPy) vs sklearn SVM (RBF kernel)
trên tập dữ liệu Chest X-Ray Images (Pneumonia).

Cách chạy:
    python main_as2.py
"""

import time
import numpy as np
from pathlib import Path

from data_loader   import find_split_dir, load_split
from preprocessing import fit_transform_combined
from svm_numpy     import SoftMarginSVM
from svm_sklearn   import build_sklearn_svm, train_sklearn_svm
from metrics       import print_metrics
from visualization import (
    plot_sample_images,
    plot_loss,
    plot_confusion_matrices_comparison,
    plot_metrics_comparison,
)

# ── 1. CẤU HÌNH ───────────────────────────────────────────────────────────────
DATA_ROOT    = Path(r"C:\Users\DELL\Downloads\archive (1)")
IMG_SIZE     = 128
N_COMPONENTS = 150
C            = 1.0   # dùng chung cho cả 2 model

# ── 2. TÌM ĐƯỜNG DẪN DỮ LIỆU ─────────────────────────────────────────────────
train_dir = find_split_dir(DATA_ROOT, "train")
val_dir   = find_split_dir(DATA_ROOT, "val")
test_dir  = find_split_dir(DATA_ROOT, "test")

print(f"✓ Dataset root : {DATA_ROOT.resolve()}")
print(f"  train → {train_dir}")
print(f"  val   → {val_dir}")
print(f"  test  → {test_dir}")

# ── 3. LOAD DỮ LIỆU ───────────────────────────────────────────────────────────
print("\nLoading …")
t0 = time.time()
X_train, y_train = load_split(train_dir, IMG_SIZE)
X_val,   y_val   = load_split(val_dir,   IMG_SIZE)
X_test,  y_test  = load_split(test_dir,  IMG_SIZE)

# Gộp train + val để train sklearn SVM
X_combined = np.concatenate([X_train, X_val])
y_combined = np.concatenate([y_train, y_val])
n_train    = len(X_train)

print(f"Done in {time.time() - t0:.1f}s")
print(f"  Train+Val {X_combined.shape}  Test {X_test.shape}")

# ── 4. HIỂN THỊ MẪU ẢNH ──────────────────────────────────────────────────────
plot_sample_images(X_train, y_train, img_size=IMG_SIZE)

# ── 5. TIỀN XỬ LÝ: StandardScaler + PCA ─────────────────────────────────────
X_train_pca, X_val_pca, X_test_pca, scaler, pca = fit_transform_combined(
    X_combined, X_test, n_train=n_train, n_components=N_COMPONENTS
)
# X_combined_pca để train sklearn (toàn bộ train+val)
import numpy as _np
X_combined_pca = _np.concatenate([X_train_pca, X_val_pca])

# ── 6A. HUẤN LUYỆN CUSTOM SVM ─────────────────────────────────────────────────
print("\n[Custom SVM] Training …")
custom_svm = SoftMarginSVM(C=C, lr=0.01, n_epochs=50, batch_size=64, lr_decay=0.95)
t0 = time.time()
custom_svm.fit(X_train_pca, y_train, X_val_pca, y_val)
custom_time = time.time() - t0
print(f"  Time: {custom_time:.1f}s")

plot_loss(custom_svm.train_losses, custom_svm.val_losses,
          title="Loss — Custom SVM (NumPy + Mini-batch SGD)")

# ── 6B. HUẤN LUYỆN SKLEARN SVM ────────────────────────────────────────────────
sklearn_model = build_sklearn_svm(C=C, kernel="rbf")
sklearn_model, sklearn_time = train_sklearn_svm(
    sklearn_model, X_combined_pca, y_combined
)
print("\n[sklearn SVM] Không có loss curve (dùng thuật toán SMO nội bộ).")

# ── 7. ĐÁNH GIÁ ───────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("  CUSTOM SVM (NumPy + SGD)")
print("=" * 50)
custom_m = print_metrics(y_test, custom_svm.predict(X_test_pca),
                         "Custom SVM — Test")

print("\n" + "=" * 50)
print("  SKLEARN SVM (RBF Kernel)")
print("=" * 50)
sklearn_m = print_metrics(y_test, sklearn_model.predict(X_test_pca),
                          "sklearn SVM — Test")

# ── 8. BIỂU ĐỒ SO SÁNH ───────────────────────────────────────────────────────
plot_confusion_matrices_comparison(
    custom_m, sklearn_m,
    title1="Custom SVM (NumPy + SGD)",
    title2="sklearn SVM (RBF)",
)
plot_metrics_comparison(
    custom_m, sklearn_m,
    label1="Custom SVM (NumPy+SGD)",
    label2="sklearn SVM (RBF)",
    title="Custom SVM vs sklearn SVM — Test Set Comparison",
)

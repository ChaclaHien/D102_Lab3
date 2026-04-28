"""
main_as1.py
===========
Assignment 1 – Soft-Margin SVM tự cài đặt bằng NumPy + Mini-batch SGD
trên tập dữ liệu Chest X-Ray Images (Pneumonia).

Cách chạy:
    python main_as1.py
"""

import time
import numpy as np
from pathlib import Path

from data_loader   import find_split_dir, load_split
from preprocessing import fit_transform_pipeline
from svm_numpy     import SoftMarginSVM
from metrics       import print_metrics
from visualization import (
    plot_sample_images,
    plot_loss,
    plot_confusion_matrix,
    plot_metrics_bar,
)

# ── 1. CẤU HÌNH ───────────────────────────────────────────────────────────────
DATA_ROOT    = Path(r"C:\Users\DELL\Downloads\archive (1)")
IMG_SIZE     = 128
N_COMPONENTS = 150
C            = 1.0
LR           = 0.01
N_EPOCHS     = 50
BATCH_SIZE   = 64
LR_DECAY     = 0.95

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
print(f"Done in {time.time() - t0:.1f}s")
print(f"  Train {X_train.shape}  Val {X_val.shape}  Test {X_test.shape}")

# ── 4. HIỂN THỊ MẪU ẢNH ──────────────────────────────────────────────────────
plot_sample_images(X_train, y_train, img_size=IMG_SIZE)

# ── 5. TIỀN XỬ LÝ: StandardScaler + PCA ─────────────────────────────────────
X_train_pca, X_val_pca, X_test_pca, scaler, pca = fit_transform_pipeline(
    X_train, X_val, X_test, n_components=N_COMPONENTS
)

# ── 6. HUẤN LUYỆN ─────────────────────────────────────────────────────────────
print("\nTraining Soft-Margin SVM …")
svm = SoftMarginSVM(C=C, lr=LR, n_epochs=N_EPOCHS,
                    batch_size=BATCH_SIZE, lr_decay=LR_DECAY)
t0 = time.time()
svm.fit(X_train_pca, y_train, X_val_pca, y_val)
print(f"Time: {time.time() - t0:.1f}s")

# ── 7. LOSS CURVE ─────────────────────────────────────────────────────────────
plot_loss(svm.train_losses, svm.val_losses, title="Loss Curve")

# ── 8. ĐÁNH GIÁ ───────────────────────────────────────────────────────────────
y_train_pred = svm.predict(X_train_pca)
y_val_pred   = svm.predict(X_val_pca)
y_test_pred  = svm.predict(X_test_pca)

train_m = print_metrics(y_train, y_train_pred, "Train")
val_m   = print_metrics(y_val,   y_val_pred,   "Val")
test_m  = print_metrics(y_test,  y_test_pred,  "Test")

# ── 9. BIỂU ĐỒ ────────────────────────────────────────────────────────────────
plot_confusion_matrix(test_m, title="Confusion Matrix — Test Set")
plot_metrics_bar(train_m, val_m, test_m,
                 title="Soft-Margin SVM — Performance Summary")

"""
visualization.py
================
Các hàm vẽ đồ thị: loss curve, confusion matrix, metrics bar chart,
và hiển thị mẫu ảnh.
"""

import numpy as np
import matplotlib.pyplot as plt


# ── Loss Curve ────────────────────────────────────────────────────────────────

def plot_loss(
    train_losses: list,
    val_losses: list | None = None,
    title: str = "Loss - PCA Features + Mini-batch SGD",
) -> None:
    """
    Vẽ đường cong loss theo epoch.

    Parameters
    ----------
    train_losses : list  – loss trên tập train theo từng epoch
    val_losses   : list  – loss trên tập val (tuỳ chọn)
    title        : str   – tiêu đề biểu đồ
    """
    epochs = np.arange(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(epochs, train_losses, color="red", linewidth=1.2, label="Train Loss")
    if val_losses and len(val_losses) == len(train_losses):
        ax.plot(
            epochs, val_losses,
            color="steelblue", linewidth=1.2, linestyle="--", label="Val Loss",
        )
        ax.legend(fontsize=11)

    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Epochs", fontsize=12)
    ax.set_ylabel("Loss",   fontsize=12)
    ax.set_xlim(0, len(train_losses))
    ax.set_ylim(bottom=0)
    ax.grid(True, color="lightgray", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()


# ── Confusion Matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(m: dict, title: str = "Confusion Matrix — Test Set") -> None:
    """
    Vẽ confusion matrix cho một mô hình.

    Parameters
    ----------
    m     : dict – kết quả từ metrics.compute_metrics() (có key tp, fp, fn, tn)
    title : str  – tiêu đề biểu đồ
    """
    cm = np.array([[m["tp"], m["fn"]], [m["fp"], m["tn"]]])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, cm[i, j],
                ha="center", va="center", fontsize=16, fontweight="bold",
                color="white" if cm[i, j] > cm.max() * 0.6 else "black",
            )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred POS\n(Pneumonia)", "Pred NEG\n(Normal)"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["True POS\n(Pneumonia)", "True NEG\n(Normal)"])
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrices_comparison(
    m1: dict, m2: dict,
    title1: str = "Custom SVM (NumPy + SGD)",
    title2: str = "sklearn SVM (RBF)",
) -> None:
    """
    Vẽ 2 confusion matrix cạnh nhau để so sánh 2 mô hình (Assignment 2).

    Parameters
    ----------
    m1, m2         : dict – kết quả từ metrics.compute_metrics()
    title1, title2 : str  – tên của từng mô hình
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, m, title in zip(axes, [m1, m2], [title1, title2]):
        cm = np.array([[m["tp"], m["fn"]], [m["fp"], m["tn"]]])
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, cm[i, j],
                    ha="center", va="center", fontsize=15, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() * 0.6 else "black",
                )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pred POS\n(Pneumonia)", "Pred NEG\n(Normal)"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["True POS\n(Pneumonia)", "True NEG\n(Normal)"])
        ax.set_title(f"Confusion Matrix\n{title}")
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()


# ── Metrics Bar Chart ─────────────────────────────────────────────────────────

def plot_metrics_bar(
    train_m: dict,
    val_m: dict,
    test_m: dict,
    title: str = "Soft-Margin SVM — Performance Summary",
) -> None:
    """
    Vẽ bar chart so sánh Accuracy / Precision / Recall / F1
    trên 3 tập Train / Val / Test (Assignment 1).

    Parameters
    ----------
    train_m, val_m, test_m : dict – kết quả từ metrics.compute_metrics()
    title                  : str  – tiêu đề biểu đồ
    """
    metrics  = ["Accuracy", "Precision", "Recall", "F1"]
    keys     = ["acc", "prec", "rec", "f1"]
    train_v  = [train_m[k] for k in keys]
    val_v    = [val_m[k]   for k in keys]
    test_v   = [test_m[k]  for k in keys]

    x  = np.arange(len(metrics))
    bw = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - bw, train_v, bw, label="Train",      color="steelblue")
    b2 = ax.bar(x,      val_v,   bw, label="Validation", color="orange")
    b3 = ax.bar(x + bw, test_v,  bw, label="Test",       color="tomato")

    for bars in [b1, b2, b3]:
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_metrics_comparison(
    m1: dict,
    m2: dict,
    label1: str = "Custom SVM (NumPy+SGD)",
    label2: str = "sklearn SVM (RBF)",
    title: str = "Custom SVM vs sklearn SVM — Test Set Comparison",
) -> None:
    """
    Vẽ bar chart so sánh metrics giữa 2 mô hình (Assignment 2).

    Parameters
    ----------
    m1, m2         : dict – kết quả từ metrics.compute_metrics()
    label1, label2 : str  – tên mô hình
    title          : str  – tiêu đề biểu đồ
    """
    metrics  = ["Accuracy", "Precision", "Recall", "F1"]
    keys     = ["acc", "prec", "rec", "f1"]
    v1 = [m1[k] for k in keys]
    v2 = [m2[k] for k in keys]

    x  = np.arange(len(metrics))
    bw = 0.3
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - bw / 2, v1, bw, label=label1, color="steelblue")
    b2 = ax.bar(x + bw / 2, v2, bw, label=label2, color="tomato")

    for bars in [b1, b2]:
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center", fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    plt.show()


# ── Sample Images ─────────────────────────────────────────────────────────────

def plot_sample_images(
    X: np.ndarray,
    y: np.ndarray,
    img_size: int = 128,
    n_per_class: int = 4,
) -> None:
    """
    Hiển thị một số ảnh mẫu từ tập dữ liệu.

    Parameters
    ----------
    X           : np.ndarray – dữ liệu ảnh đã flatten (N, img_size²)
    y           : np.ndarray – nhãn (+1 / -1)
    img_size    : int        – chiều cao/rộng ảnh gốc
    n_per_class : int        – số ảnh hiển thị mỗi nhãn
    """
    fig, axes = plt.subplots(2, n_per_class, figsize=(3 * n_per_class, 6))

    for ax, idx in zip(axes[0], np.where(y == -1)[0][:n_per_class]):
        ax.imshow(X[idx].reshape(img_size, img_size), cmap="gray")
        ax.set_title("NORMAL", fontsize=9)
        ax.axis("off")

    for ax, idx in zip(axes[1], np.where(y == 1)[0][:n_per_class]):
        ax.imshow(X[idx].reshape(img_size, img_size), cmap="gray")
        ax.set_title("PNEUMONIA", fontsize=9, color="tomato")
        ax.axis("off")

    plt.suptitle(f"Sample Images ({img_size}×{img_size} grayscale)")
    plt.tight_layout()
    plt.show()

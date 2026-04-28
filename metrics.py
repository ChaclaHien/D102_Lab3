"""
metrics.py
==========
Tính và in các chỉ số đánh giá: Accuracy, Precision, Recall, F1-Score.
Nhãn dương = PNEUMONIA (+1), nhãn âm = NORMAL (-1).
"""

import numpy as np


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Tính confusion matrix và các chỉ số đánh giá.

    Parameters
    ----------
    y_true : np.ndarray – nhãn thực tế (+1 / -1)
    y_pred : np.ndarray – nhãn dự đoán (+1 / -1)

    Returns
    -------
    dict với các key: acc, prec, rec, f1, tp, fp, fn, tn
    """
    tp = int(((y_pred ==  1) & (y_true ==  1)).sum())
    fp = int(((y_pred ==  1) & (y_true == -1)).sum())
    fn = int(((y_pred == -1) & (y_true ==  1)).sum())
    tn = int(((y_pred == -1) & (y_true == -1)).sum())

    acc  = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return dict(acc=acc, prec=prec, rec=rec, f1=f1,
                tp=tp, fp=fp, fn=fn, tn=tn)


def print_metrics(y_true: np.ndarray, y_pred: np.ndarray, split: str = "") -> dict:
    """
    Tính, in và trả về các chỉ số đánh giá.

    Parameters
    ----------
    y_true : np.ndarray – nhãn thực tế
    y_pred : np.ndarray – nhãn dự đoán
    split  : str        – tên tập dữ liệu (dùng để in tiêu đề)

    Returns
    -------
    dict – xem compute_metrics()
    """
    m = compute_metrics(y_true, y_pred)

    print(f"\n{'═' * 42}")
    print(f"  {split}")
    print(f"{'═' * 42}")
    print(f"  Accuracy  : {m['acc']:.4f}")
    print(f"  Precision : {m['prec']:.4f}")
    print(f"  Recall    : {m['rec']:.4f}")
    print(f"  F1-Score  : {m['f1']:.4f}")
    print(f"  Confusion Matrix (positive = PNEUMONIA)")
    print(f"              Pred POS   Pred NEG")
    print(f"  True POS :   {m['tp']:5d}      {m['fn']:5d}")
    print(f"  True NEG :   {m['fp']:5d}      {m['tn']:5d}")
    print(f"{'═' * 42}")

    return m

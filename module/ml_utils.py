import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Dữ liệu
# ---------------------------------------------------------------------------
def load_data(path, drop_duplicates=True):
    """Đọc dữ liệu và tùy chọn loại bỏ các dòng trùng lặp."""
    df = pd.read_csv(path)
    n0 = len(df)
    if drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
    print(
        f"Đọc {n0:,} dòng -> {len(df):,} dòng sau khi bỏ trùng | "
        f"fraud = {int(df['Class'].sum())} ({df['Class'].mean() * 100:.3f}%)"
    )
    return df


def split_time(df, train_frac=0.6, val_frac=0.2):
    """Chia theo thời gian: train -> validation -> test."""
    df = df.sort_values("Time").reset_index(drop=True)
    i = int(len(df) * train_frac)
    j = int(len(df) * (train_frac + val_frac))
    return df.iloc[:i], df.iloc[i:j], df.iloc[j:]


def split_stratified(df, seed):
    """Chia ngẫu nhiên có phân tầng theo Class: 60/20/20."""
    train_val, test = train_test_split(df, test_size=0.2, stratify=df["Class"], random_state=seed)
    train, val = train_test_split(
        train_val, test_size=0.25, stratify=train_val["Class"], random_state=seed
    )
    return train, val, test


def get_xy(part, features):
    """Tách DataFrame thành ma trận đặc trưng X và nhãn y."""
    return part[features].values, part["Class"].values


# ---------------------------------------------------------------------------
# Đánh giá
# ---------------------------------------------------------------------------
def best_threshold(y, score):
    """Chọn ngưỡng có F1 cao nhất (dùng trên tập validation)."""
    p, r, thresholds = precision_recall_curve(y, score)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])


def evaluate(y, score, thr):
    """Tính các metric và confusion matrix tại ngưỡng đã chọn."""
    pred = (score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(thr),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def run_eval(train_fn, train, val, test, features, seed):
    """Huấn luyện, chọn threshold trên validation, đánh giá trên test và đo thời gian.

    train_fn(X_tr, y_tr, X_va, y_va, seed) -> (model, score_fn, train_time_s)
    """
    X_tr, y_tr = get_xy(train, features)
    X_va, y_va = get_xy(val, features)
    X_te, y_te = get_xy(test, features)

    model, score_fn, train_time = train_fn(X_tr, y_tr, X_va, y_va, seed)

    # Threshold chỉ được chọn trên validation.
    thr = best_threshold(y_va, score_fn(X_va))

    # Thời gian suy luận theo batch (giây / mẫu * 1e6 = ms cho mỗi 1000 giao dịch).
    t0 = time.perf_counter()
    s_te = score_fn(X_te)
    batch_time = time.perf_counter() - t0

    # Độ trễ khi suy luận từng giao dịch (giống lúc gọi API).
    n_single = min(100, len(X_te))
    t0 = time.perf_counter()
    for k in range(n_single):
        score_fn(X_te[k : k + 1])
    single_ms = (time.perf_counter() - t0) / n_single * 1000

    metrics = {
        **evaluate(y_te, s_te, thr),
        "train_time_s": float(train_time),
        "infer_ms_per_1k": float(batch_time / len(X_te) * 1e6),
        "latency_ms_single": float(single_ms),
    }
    art = {"model": model, "score_fn": score_fn, "y_te": y_te, "s_te": s_te, "thr": thr}
    return metrics, art


# ---------------------------------------------------------------------------
# Biểu đồ
# ---------------------------------------------------------------------------
def plot_eval(art, title, path):
    """Vẽ Precision-Recall, ROC và confusion matrix trên tập test."""
    y, s, thr = art["y_te"], art["s_te"], art["thr"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # Precision-Recall
    p, r, _ = precision_recall_curve(y, s)
    axes[0].plot(r, p, label=f"AP = {average_precision_score(y, s):.3f}")
    axes[0].axhline(y.mean(), ls="--", c="gray", label="Baseline (tỉ lệ fraud)")
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall (test)")
    axes[0].legend()

    # ROC
    fpr, tpr, _ = roc_curve(y, s)
    axes[1].plot(fpr, tpr, label=f"AUC = {roc_auc_score(y, s):.3f}")
    axes[1].plot([0, 1], [0, 1], ls="--", c="gray")
    axes[1].set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC (test)")
    axes[1].legend()

    # Confusion matrix
    cm = confusion_matrix(y, (s >= thr).astype(int), labels=[0, 1])
    axes[2].imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        color = "white" if v > cm.max() / 2 else "black"
        axes[2].text(j, i, f"{v:,}", ha="center", va="center", fontsize=12, color=color)
    labels = ["Legit", "Fraud"]
    axes[2].set(xticks=[0, 1], yticks=[0, 1], xticklabels=labels, yticklabels=labels)
    axes[2].set(xlabel="Predicted", ylabel="Actual", title="Confusion matrix (test)")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.show()
    plt.close(fig)
"""
Fruit Freshness Classification — Класичний ML підхід.
Метод 2: Ручне вилучення ознак (HOG + LBP + Color Histogram) + класичні класифікатори.

Класифікатори для порівняння (аналогічно до PDF MaterialRecognition):
    - K-Nearest Neighbors (KNN)
    - Linear SVM
    - RBF SVM
    - Decision Tree
    - Random Forest
    - AdaBoost
    - Naive Bayes

Ознаки:
    - HOG  (Histogram of Oriented Gradients) — форма і контури
    - LBP  (Local Binary Patterns)           — текстура поверхні
    - Color Histogram (HSV)                  — колірний розподіл
"""

import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from skimage.feature import hog, local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import (auc, classification_report, confusion_matrix,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")


DATA_ROOT   = Path("dataset/Fruit Freshness Dataset/Fruit Freshness Dataset")
IMG_SIZE    = 128
VALID_EXT   = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CLASS_NAMES = ["Fresh", "Rotten"]
SEED        = 42


def collect_samples(root: Path) -> tuple[list, list]:
    paths, labels = [], []
    label_map = {"fresh": 0, "rotten": 1}
    for fruit_dir in sorted(root.iterdir()):
        if not fruit_dir.is_dir():
            continue
        for quality_dir in sorted(fruit_dir.iterdir()):
            if not quality_dir.is_dir():
                continue
            label = label_map.get(quality_dir.name.lower())
            if label is None:
                continue
            for f in quality_dir.iterdir():
                if f.suffix.lower() in VALID_EXT:
                    paths.append(str(f))
                    labels.append(label)
    print(f"Знайдено: {len(paths)} зображень | "
          f"Fresh={labels.count(0)}, Rotten={labels.count(1)}")
    return paths, labels


def extract_hog(img_gray: np.ndarray) -> np.ndarray:
    """
    HOG — гістограма орієнтованих градієнтів.
    Захоплює форму, краї, контури (плями, зморшки гнилого фрукта).
    """
    features, _ = hog(
        img_gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        visualize=True,
        feature_vector=True,
    )
    return features


def extract_lbp(img_gray: np.ndarray) -> np.ndarray:
    """
    LBP — локальні бінарні паттерни.
    Захоплює мікротекстуру поверхні (суха/волога шкірка, пліснява).
    """
    radius, n_points = 3, 24
    lbp = local_binary_pattern(img_gray, n_points, radius, method="uniform")
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins,
                           range=(0, n_bins), density=True)
    return hist


def extract_color_hist(img_hsv: np.ndarray) -> np.ndarray:
    """
    Колірна гістограма у просторі HSV.
    Захоплює зміни кольору (жовтіння, потемніння, бурі плями).
    H: 18 бінів, S: 16 бінів, V: 16 бінів → 50 ознак
    """
    h_hist, _ = np.histogram(img_hsv[:, :, 0], bins=18,
                              range=(0, 180), density=True)
    s_hist, _ = np.histogram(img_hsv[:, :, 1], bins=16,
                              range=(0, 256), density=True)
    v_hist, _ = np.histogram(img_hsv[:, :, 2], bins=16,
                              range=(0, 256), density=True)
    return np.concatenate([h_hist, s_hist, v_hist])


def extract_features(image_path: str) -> np.ndarray:
    """Повний вектор ознак для одного зображення."""
    img = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img_np   = np.array(img)
    img_gray = np.array(img.convert("L"))

    img_f    = img_np.astype(np.float32) / 255.0
    r, g, b  = img_f[..., 0], img_f[..., 1], img_f[..., 2]
    maxc     = np.max(img_f, axis=2)
    minc     = np.min(img_f, axis=2)
    diff     = maxc - minc + 1e-8
    h        = np.zeros_like(maxc)
    mask_r   = maxc == r
    mask_g   = maxc == g
    mask_b   = maxc == b
    h[mask_r] = ((g[mask_r] - b[mask_r]) / diff[mask_r]) % 6
    h[mask_g] = (b[mask_g] - r[mask_g]) / diff[mask_g] + 2
    h[mask_b] = (r[mask_b] - g[mask_b]) / diff[mask_b] + 4
    h = (h / 6.0 * 180).clip(0, 180)
    s = np.where(maxc > 0, diff / maxc * 255, 0)
    v = maxc * 255
    img_hsv = np.stack([h, s, v], axis=2)

    hog_feat   = extract_hog(img_gray)
    lbp_feat   = extract_lbp(img_gray)
    color_feat = extract_color_hist(img_hsv)

    return np.concatenate([hog_feat, lbp_feat, color_feat])


def build_feature_matrix(paths: list, labels: list) -> tuple[np.ndarray, np.ndarray]:
    print("Вилучення ознак...", flush=True)
    X = []
    for i, p in enumerate(paths):
        X.append(extract_features(p))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(paths)}", flush=True)
    X = np.array(X)
    y = np.array(labels)
    print(f"Матриця ознак: {X.shape}  "
          f"(зображень={X.shape[0]}, ознак={X.shape[1]})")
    return X, y


def get_classifiers() -> dict:
    return {
        "KNN":           KNeighborsClassifier(n_neighbors=7, weights="distance",
                                              metric="euclidean", n_jobs=-1),
        "Linear SVM":    SVC(kernel="linear", C=1.0, probability=True,
                             random_state=SEED, max_iter=5000),
        "RBF SVM":       SVC(kernel="rbf",    C=10.0, gamma="scale",
                             probability=True, random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=None, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=None,
                                                random_state=SEED, n_jobs=-1),
        "AdaBoost":      AdaBoostClassifier(n_estimators=100, learning_rate=0.5,
                                            random_state=SEED),
        "Naive Bayes":   GaussianNB(),
    }


def train_and_evaluate(clf, X_train, y_train, X_test, y_test, name: str,
                       index: int = 0, total: int = 7) -> dict:
    print(f"\n[{index}/{total}] Навчання: {name}...", flush=True)
    t_start = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t_start

    t_inf = time.time()
    preds = clf.predict(X_test)
    inference_time = (time.time() - t_inf) / len(y_test) * 1000

    probs = clf.predict_proba(X_test)[:, 1]

    acc = (preds == y_test).mean()
    report = classification_report(y_test, preds, target_names=CLASS_NAMES,
                                   output_dict=True)
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  Accuracy:        {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  AUC:             {roc_auc:.4f}")
    print(f"  Час навчання:    {train_time:.2f}s")
    print(f"  Inference:       {inference_time:.3f} ms/зображення")
    print(classification_report(y_test, preds, target_names=CLASS_NAMES))

    return {
        "name":                   name,
        "accuracy":               float(acc),
        "precision_fresh":        report["Fresh"]["precision"],
        "recall_fresh":           report["Fresh"]["recall"],
        "f1_fresh":               report["Fresh"]["f1-score"],
        "precision_rotten":       report["Rotten"]["precision"],
        "recall_rotten":          report["Rotten"]["recall"],
        "f1_rotten":              report["Rotten"]["f1-score"],
        "auc":                    float(roc_auc),
        "train_time_s":           float(train_time),
        "inference_ms_per_image": float(inference_time),
        "fpr":                    fpr.tolist(),
        "tpr":                    tpr.tolist(),
        "preds":                  preds.tolist(),
        "labels":                 y_test.tolist(),
    }


def plot_all_confusion(results: list):
    n = len(results)
    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    axes = axes.flat
    for i, res in enumerate(results):
        cm = confusion_matrix(res["labels"], res["preds"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        axes[i].set_title(f"{res['name']}\nAcc={res['accuracy']:.3f}", fontsize=10)
        axes[i].set_xlabel("Передбачений"); axes[i].set_ylabel("Справжній")
    for j in range(n, 9):
        axes[j].axis("off")
    plt.suptitle("Матриці плутанини — класичні методи (7 алгоритмів)", fontsize=13)
    plt.tight_layout()
    plt.savefig("classical_confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Збережено: classical_confusion_matrices.png")


def plot_all_roc(results: list):
    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    for i, res in enumerate(results):
        plt.plot(res["fpr"], res["tpr"], lw=2, color=cmap(i),
                 label=f"{res['name']}  AUC={res['auc']:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("FPR (False Positive Rate)", fontsize=11)
    plt.ylabel("TPR (True Positive Rate)", fontsize=11)
    plt.title("ROC-криві — класичні методи (7 алгоритмів)", fontsize=12)
    plt.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    plt.savefig("classical_roc_curves.png", dpi=150)
    plt.close()
    print("Збережено: classical_roc_curves.png")


def plot_accuracy_bar(results: list):
    """Порівняння точності всіх 7 алгоритмів (стиль як у слайдах курсової)."""
    names = [r["name"] for r in results]
    accs  = [r["accuracy"] * 100 for r in results]
    recalls = [r["recall_rotten"] * 100 for r in results]

    x = np.arange(len(names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, accs,   width, label="Accuracy (%)",       color="#4c72b0")
    bars2 = ax.bar(x + width / 2, recalls, width, label="Recall Rotten (%)", color="#dd8452")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Відсоток (%)", fontsize=11)
    ax.set_title("Порівняння методів: Accuracy та Recall(Rotten)", fontsize=13)
    ax.legend(fontsize=10)
    ax.axhline(y=90, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("classical_accuracy_bar.png", dpi=150)
    plt.close()
    print("Збережено: classical_accuracy_bar.png")


def plot_feature_importance(X_train_scaled: np.ndarray, y_train: np.ndarray):
    """Важливість груп ознак (HOG / LBP / Color) для Random Forest.

    Використовуємо scaled ознаки БЕЗ PCA, бо після PCA компоненти —
    це лінійні комбінації оригінальних ознак і їх не можна інтерпретувати
    як HOG/LBP/Color.
    """
    rf_imp = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    rf_imp.fit(X_train_scaled, y_train)
    importances = rf_imp.feature_importances_

    gray128 = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    hog_n   = len(extract_hog(gray128))
    lbp_n   = 26
    color_n = 50

    groups = {
        f"HOG ({hog_n} ознак)":    importances[:hog_n].sum(),
        f"LBP ({lbp_n} ознак)":    importances[hog_n:hog_n + lbp_n].sum(),
        f"Color ({color_n} ознак)": importances[hog_n + lbp_n:].sum(),
    }
    plt.figure(figsize=(6, 4))
    bars = plt.bar(groups.keys(), groups.values(),
                   color=["#4daf4a", "#377eb8", "#e41a1c"])
    plt.ylabel("Сумарна важливість")
    plt.title("Внесок груп ознак (Random Forest)")
    for bar, val in zip(bars, groups.values()):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.close()
    print("Збережено: feature_importance.png")


def main():
    np.random.seed(SEED)

    paths, labels = collect_samples(DATA_ROOT)

    t_feat = time.time()
    X, y   = build_feature_matrix(paths, labels)
    feat_time = time.time() - t_feat
    print(f"Час вилучення ознак: {feat_time:.1f}s")

    X_tr_raw, X_ts_raw, y_tr, y_ts = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED)
    print(f"Спліт: train={len(y_tr)}, test={len(y_ts)}")

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr_raw)
    X_ts_sc = scaler.transform(X_ts_raw)

    pca = PCA(n_components=0.97, random_state=SEED)
    X_tr = pca.fit_transform(X_tr_sc)
    X_ts = pca.transform(X_ts_sc)
    print(f"PCA: {X.shape[1]} → {X_tr.shape[1]} компонент "
          f"(97% дисперсії збережено)")

    classifiers = get_classifiers()
    total = len(classifiers)
    results = []
    for i, (name, clf) in enumerate(classifiers.items(), start=1):
        res = train_and_evaluate(clf, X_tr, y_tr, X_ts, y_ts, name,
                                 index=i, total=total)
        results.append(res)

    print(f"\n{'='*75}")
    print(f"{'Метод':<18} {'Accuracy':>9} {'Recall(Rotten)':>15} {'AUC':>7} {'Inf(ms)':>9}")
    print(f"{'─'*75}")
    for r in results:
        print(f"{r['name']:<18} {r['accuracy']*100:>8.1f}%"
              f" {r['recall_rotten']*100:>13.1f}%"
              f" {r['auc']:>7.3f}"
              f" {r['inference_ms_per_image']:>8.3f}")
    print(f"{'='*75}")

    best_acc = max(results, key=lambda r: r["accuracy"])
    best_rec = max(results, key=lambda r: r["recall_rotten"])
    print(f"Найкращий за Accuracy:       {best_acc['name']}  ({best_acc['accuracy']*100:.1f}%)")
    print(f"Найкращий за Recall(Rotten): {best_rec['name']}  ({best_rec['recall_rotten']*100:.1f}%)")

    print("\nГенерація графіків...")
    plot_all_confusion(results)
    plot_all_roc(results)
    plot_accuracy_bar(results)
    plot_feature_importance(X_tr_sc, y_tr)

    output = {
        "feature_extraction_time_s": float(feat_time),
        "feature_dim_original":      int(X.shape[1]),
        "feature_dim_pca":           int(X_tr.shape[1]),
        "train_images":              int(len(y_tr)),
        "test_images":               int(len(y_ts)),
        "classifiers":               results,
    }
    with open("results_classical.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print("\nРезультати збережено → results_classical.json")

    print("\nГотово! Збережені файли:")
    for fname in ["results_classical.json", "classical_confusion_matrices.png",
                  "classical_roc_curves.png", "classical_accuracy_bar.png",
                  "feature_importance.png"]:
        print(f"  {fname}")


if __name__ == "__main__":
    main()

"""
Порівняння класичних методів розпізнавання якості фруктів.

Читає: results_classical.json  (вихід method2_classical.py)
Методи: KNN, Linear SVM, RBF SVM, Decision Tree, Random Forest, AdaBoost, Naive Bayes

Запуск: python compare.py
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


with open("results_classical.json", encoding="utf-8") as f:
    cl = json.load(f)

methods = []
for c in cl["classifiers"]:
    methods.append({
        "name":                   c["name"],
        "name_wrap":              c["name"].replace(" ", "\n"),
        "accuracy":               c["accuracy"],
        "f1_fresh":               c["f1_fresh"],
        "f1_rotten":              c["f1_rotten"],
        "recall_rotten":          c["recall_rotten"],
        "auc":                    c["auc"],
        "inference_ms_per_image": c["inference_ms_per_image"],
        "train_time_s":           c["train_time_s"],
    })

names_wrap = [m["name_wrap"]              for m in methods]
names      = [m["name"]                   for m in methods]
accuracies = [m["accuracy"]               for m in methods]
aucs       = [m["auc"]                    for m in methods]
recall_rot = [m["recall_rotten"]          for m in methods]
f1_fresh   = [m["f1_fresh"]               for m in methods]
f1_rotten  = [m["f1_rotten"]             for m in methods]
inf_times  = [m["inference_ms_per_image"] for m in methods]

n = len(methods)
x = np.arange(n)
COLOR = "#377eb8"


print("\n" + "=" * 95)
print(f"{'Метод':<18} {'Accuracy':>9} {'AUC':>7} {'F1 Fresh':>9} "
      f"{'F1 Rotten':>10} {'Recall Rotten':>14} {'Inf ms':>8}")
print("=" * 95)
for m in methods:
    print(f"{m['name']:<18} {m['accuracy']:>9.4f} {m['auc']:>7.4f} "
          f"{m['f1_fresh']:>9.4f} {m['f1_rotten']:>10.4f} "
          f"{m['recall_rotten']:>14.4f} {m['inference_ms_per_image']:>8.3f}")
print("=" * 95)

best_acc = max(methods, key=lambda m: m["accuracy"])
best_auc = max(methods, key=lambda m: m["auc"])
best_rec = max(methods, key=lambda m: m["recall_rotten"])
print(f"\nНайкраща Accuracy:       {best_acc['name']}  → {best_acc['accuracy']:.4f}")
print(f"Найкращий AUC:           {best_auc['name']}  → {best_auc['auc']:.4f}")
print(f"Найкращий Recall(Rotten):{best_rec['name']}  → {best_rec['recall_rotten']:.4f}")


fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Порівняння класичних методів визначення якості фруктів",
             fontsize=14, fontweight="bold")

def annotate_bars(ax, bars, fmt=".3f"):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():{fmt}}", ha="center", fontsize=8, fontweight="bold")

ax = axes[0, 0]
bars = ax.bar(x, accuracies, color=COLOR, edgecolor="black", linewidth=0.7)
ax.axhline(max(accuracies), color="gold", lw=1.5, ls="--", alpha=0.7)
ax.set_xticks(x); ax.set_xticklabels(names_wrap, fontsize=8)
ax.set_ylabel("Accuracy"); ax.set_title("Точність (Accuracy)")
ax.set_ylim(0.5, 1.05)
annotate_bars(ax, bars)

ax = axes[0, 1]
bars = ax.bar(x, aucs, color=COLOR, edgecolor="black", linewidth=0.7)
ax.set_xticks(x); ax.set_xticklabels(names_wrap, fontsize=8)
ax.set_ylabel("AUC"); ax.set_title("Area Under ROC Curve (AUC)")
ax.set_ylim(0.5, 1.05)
annotate_bars(ax, bars)

ax = axes[1, 0]
width = 0.35
ax.bar(x - width/2, f1_fresh,  width, label="F1 Fresh",  color="#4daf4a", edgecolor="black", lw=0.7)
ax.bar(x + width/2, f1_rotten, width, label="F1 Rotten", color="#ff7f00", edgecolor="black", lw=0.7)
ax.set_xticks(x); ax.set_xticklabels(names_wrap, fontsize=8)
ax.set_ylabel("F1-score"); ax.set_title("F1-score по класах")
ax.set_ylim(0, 1.1); ax.legend(fontsize=9)

ax = axes[1, 1]
bars = ax.bar(x, recall_rot, color=COLOR, edgecolor="black", linewidth=0.7)
ax.axhline(1.0, color="green", lw=1.5, ls="--", alpha=0.5, label="Ідеал")
ax.set_xticks(x); ax.set_xticklabels(names_wrap, fontsize=8)
ax.set_ylabel("Recall")
ax.set_title("Recall (Rotten) — ключова метрика\n(= скільки гнилих фруктів виявлено)")
ax.set_ylim(0, 1.15)
for bar, val in zip(bars, recall_rot):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.2f}", ha="center", fontsize=9, fontweight="bold")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("comparison_metrics.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nЗбережено: comparison_metrics.png")


fig, ax = plt.subplots(figsize=(10, 5))
colors_inf = [COLOR] * n
bars = ax.barh(names_wrap[::-1], inf_times[::-1], color=colors_inf,
               edgecolor="black", linewidth=0.7)
ax.set_xlabel("Час inference (мс / зображення)")
ax.set_title("Швидкість роботи (менше = краще)")
for bar, val in zip(bars, inf_times[::-1]):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f} мс", va="center", fontsize=9)
plt.tight_layout()
plt.savefig("comparison_inference.png", dpi=150)
plt.close()
print("Збережено: comparison_inference.png")


scored = sorted(methods,
                key=lambda m: m["accuracy"] + m["auc"] + m["recall_rotten"],
                reverse=True)
top3 = scored[:3]

categories = ["Accuracy", "AUC", "F1 Fresh", "F1 Rotten", "Recall\nRotten"]
N = len(categories)
angles = [i / float(N) * 2 * np.pi for i in range(N)]
angles += angles[:1]

radar_colors = ["#e41a1c", "#377eb8", "#4daf4a"]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})

for method, color in zip(top3, radar_colors):
    vals = [method["accuracy"], method["auc"],
            method["f1_fresh"], method["f1_rotten"],
            method["recall_rotten"]]
    vals += vals[:1]
    ax.plot(angles, vals, "-o", lw=2, color=color, label=method["name"])
    ax.fill(angles, vals, alpha=0.1, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0, 1)
ax.set_title("Радарна діаграма: топ-3 класичні методи",
             pad=20, fontsize=12)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10)
plt.tight_layout()
plt.savefig("comparison_radar.png", dpi=150, bbox_inches="tight")
plt.close()
print("Збережено: comparison_radar.png")


print("\n" + "=" * 65)
print("ПІДСУМКОВИЙ ВИСНОВОК")
print("=" * 65)
print(f"""
Датасет:  {cl['train_images'] + cl['test_images']} зображень
          train={cl['train_images']}, test={cl['test_images']}
Ознаки:   {cl['feature_dim_original']} → PCA → {cl['feature_dim_pca']} компонент

{'Метод':<18} {'Accuracy':>9} {'Recall(Rotten)':>15} {'AUC':>7}
{'─'*52}""")
for m in methods:
    mark = " ←" if m["name"] == best_acc["name"] else ""
    print(f"{m['name']:<18} {m['accuracy']*100:>8.1f}%"
          f" {m['recall_rotten']*100:>13.1f}%"
          f" {m['auc']:>7.3f}{mark}")

print(f"""
Найкращий за Accuracy:       {best_acc['name']}  ({best_acc['accuracy']*100:.1f}%)
Найкращий за Recall(Rotten): {best_rec['name']}  ({best_rec['recall_rotten']*100:.1f}%)
Найкращий за AUC:            {best_auc['name']}  ({best_auc['auc']:.3f})

Топ-3 (за комплексом метрик): {', '.join(m['name'] for m in top3)}

Час вилучення ознак: {cl['feature_extraction_time_s']:.1f}s

РЕКОМЕНДАЦІЯ:
  • Найефективніший метод: {best_acc['name']}
    Accuracy={best_acc['accuracy']*100:.1f}%, Recall(Rotten)={best_acc['recall_rotten']*100:.1f}%
  • Найшвидший: {min(methods, key=lambda m: m['inference_ms_per_image'])['name']}
    ({min(inf_times):.3f} мс/зображення)
""")
print("=" * 65)
print("\nВсі графіки збережені:")
for fname in ["comparison_metrics.png", "comparison_inference.png",
              "comparison_radar.png"]:
    print(f"  {fname}")

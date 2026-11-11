# -*- coding: utf-8 -*-
"""Bài 4: tự vẽ đường ROC của logistic và rừng ngẫu nhiên."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/sinh_vien_bo_hoc.csv")
X = df.drop(columns="bo_hoc")
y = df["bo_hoc"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
mo_hinh = {
    "Logistic": make_pipeline(StandardScaler(), LogisticRegression()),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=0),
}

fig, ax = plt.subplots(figsize=(7, 5))
duong = {}
for ten, mh in mo_hinh.items():
    p = mh.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, p)
    auc = roc_auc_score(y_te, p)
    duong[ten] = (fpr, tpr, auc)
    ax.plot(fpr, tpr, linewidth=2, label=f"{ten}, AUC={auc:.4f}")
ax.plot([0, 1], [0, 1], "--", color="gray", label="Doan ngau nhien")
ax.set_xlabel("Ty le bao dong nham (FPR)")
ax.set_ylabel("Recall (TPR)")
ax.set_title("Duong ROC")
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/roc.png", dpi=160, bbox_inches="tight")
if "agg" not in plt.get_backend().lower():
    plt.show()
else:
    plt.close(fig)

luoi = np.linspace(0, 1, 1001)
tpr_lr = np.interp(luoi, duong["Logistic"][0], duong["Logistic"][1])
tpr_rf = np.interp(luoi, duong["Random Forest"][0], duong["Random Forest"][1])
chenh = tpr_lr - tpr_rf
cat_nhau = np.any(chenh[:-1] * chenh[1:] < 0)
print(f"AUC Logistic:      {duong['Logistic'][2]:.4f}")
print(f"AUC Random Forest: {duong['Random Forest'][2]:.4f}")
print("Hai duong co doan doi thu tu/cat nhau:", bool(cat_nhau))
print("\nNhan xet:")
print("- Duong co AUC cao hon nhin chung om sat goc tren ben trai hon.")
print("- Hai duong co the cat nhau o mot so vung nguong, vi mot mo hinh co the tot hon tai FPR")
print("  thap nhung kem hon tai FPR cao. AUC tom tat toan bo duong chu khong mo ta moi diem.")
print("- Da luu hinh tai figures/roc.png.")

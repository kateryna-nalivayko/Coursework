import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")
rng = np.random.default_rng(42)

DATA_ROOT   = Path("dataset/Fruit Freshness Dataset/Fruit Freshness Dataset")
IMG_SIZE    = 128
VALID_EXT   = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CLASS_NAMES = ["Fresh", "Rotten"]
SEED        = 42

def collect_samples(root: Path):
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

def rgb_to_hsv_manual(img_f: np.ndarray) -> np.ndarray:

    r, g, b = img_f[..., 0], img_f[..., 1], img_f[..., 2]
    maxc = np.max(img_f, axis=2)
    minc = np.min(img_f, axis=2)
    diff = maxc - minc + 1e-8
    h = np.zeros_like(maxc)
    mr, mg, mb = (maxc == r), (maxc == g), (maxc == b)
    h[mr] = ((g[mr] - b[mr]) / diff[mr]) % 6
    h[mg] = (b[mg] - r[mg]) / diff[mg] + 2
    h[mb] = (r[mb] - g[mb]) / diff[mb] + 4
    h = (h / 6.0 * 180).clip(0, 180)
    s = np.where(maxc > 0, diff / maxc * 255, 0)
    v = maxc * 255
    return np.stack([h, s, v], axis=2)

def extract_hog_manual(gray: np.ndarray) -> np.ndarray:

    g = gray.astype(np.float64)
    gx = np.zeros_like(g); gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    mag = np.sqrt(gx ** 2 + gy ** 2)
    ori = np.rad2deg(np.arctan2(gy, gx)) % 180.0

    cell, nbins = 16, 9
    bin_w = 180.0 / nbins
    ncy, ncx = g.shape[0] // cell, g.shape[1] // cell
    bin_idx = (np.floor(ori / bin_w).astype(int)) % nbins

    hist = np.zeros((ncy, ncx, nbins))
    for cy in range(ncy):
        for cx in range(ncx):
            m = mag[cy*cell:(cy+1)*cell, cx*cell:(cx+1)*cell]
            b = bin_idx[cy*cell:(cy+1)*cell, cx*cell:(cx+1)*cell]
            for k in range(nbins):
                hist[cy, cx, k] = m[b == k].sum()

    eps = 1e-6
    feats = []
    for by in range(ncy - 1):
        for bx in range(ncx - 1):
            block = hist[by:by+2, bx:bx+2, :].ravel()
            block = block / np.sqrt((block ** 2).sum() + eps ** 2)
            feats.append(block)
    return np.concatenate(feats)

def _shift_bilinear(img: np.ndarray, dy: float, dx: float) -> np.ndarray:

    fy, fx = int(np.floor(dy)), int(np.floor(dx))
    ty, tx = dy - fy, dx - fx
    v00 = np.roll(np.roll(img, -fy,     0), -fx,     1)
    v01 = np.roll(np.roll(img, -fy,     0), -(fx+1), 1)
    v10 = np.roll(np.roll(img, -(fy+1), 0), -fx,     1)
    v11 = np.roll(np.roll(img, -(fy+1), 0), -(fx+1), 1)
    return ((1-ty)*(1-tx)*v00 + (1-ty)*tx*v01 +
            ty*(1-tx)*v10 + ty*tx*v11)

def extract_lbp_manual(gray: np.ndarray) -> np.ndarray:

    g = gray.astype(np.float64)
    P, R = 24, 3
    H, W = g.shape
    bits = np.zeros((P, H, W))
    for p in range(P):
        theta = 2 * np.pi * p / P
        dy = -R * np.sin(theta)
        dx = R * np.cos(theta)
        bits[p] = (_shift_bilinear(g, dy, dx) >= g).astype(np.float64)

    transitions = np.abs(bits - np.roll(bits, -1, axis=0)).sum(axis=0)
    ones = bits.sum(axis=0)
    label = np.where(transitions <= 2, ones, P + 1).astype(int)

    valid = label[R:H-R, R:W-R]
    nbins = P + 2
    hist, _ = np.histogram(valid.ravel(), bins=nbins,
                           range=(0, nbins), density=True)
    return hist

def extract_color_hist_manual(hsv: np.ndarray) -> np.ndarray:

    h, _ = np.histogram(hsv[:, :, 0], bins=18, range=(0, 180), density=True)
    s, _ = np.histogram(hsv[:, :, 1], bins=16, range=(0, 256), density=True)
    v, _ = np.histogram(hsv[:, :, 2], bins=16, range=(0, 256), density=True)
    return np.concatenate([h, s, v])

def extract_features(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    rgb  = np.array(img, dtype=np.float64) / 255.0
    gray = np.array(img.convert("L"))
    hsv  = rgb_to_hsv_manual(rgb)
    return np.concatenate([
        extract_hog_manual(gray),
        extract_lbp_manual(gray),
        extract_color_hist_manual(hsv),
    ])

def build_feature_matrix(paths, labels):
    print("Вилучення ознак (вручну)...", flush=True)
    X = []
    for i, p in enumerate(paths):
        X.append(extract_features(p))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(paths)}", flush=True)
    X = np.array(X)
    y = np.array(labels)
    print(f"Матриця ознак: {X.shape}")
    return X, y

class StandardScalerManual:
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0) + 1e-8
        return self
    def transform(self, X):
        return (X - self.mean_) / self.std_
    def fit_transform(self, X):
        return self.fit(X).transform(X)

class PCAManual:

    def __init__(self, var_keep=0.97):
        self.var_keep = var_keep
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        cov = np.cov(Xc, rowvar=False)
        eigval, eigvec = np.linalg.eigh(cov)
        order = np.argsort(eigval)[::-1]
        eigval, eigvec = eigval[order], eigvec[:, order]
        ratio = np.cumsum(eigval) / np.sum(eigval)
        k = int(np.searchsorted(ratio, self.var_keep) + 1)
        self.components_ = eigvec[:, :k]
        self.n_components_ = k
        return self
    def transform(self, X):
        return (X - self.mean_) @ self.components_
    def fit_transform(self, X):
        return self.fit(X).transform(X)

def train_test_split_manual(X, y, test_size=0.30, seed=SEED):

    rs = np.random.default_rng(seed)
    tr_idx, ts_idx = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rs.shuffle(idx)
        n_ts = int(round(len(idx) * test_size))
        ts_idx.extend(idx[:n_ts])
        tr_idx.extend(idx[n_ts:])
    tr_idx = np.array(tr_idx); ts_idx = np.array(ts_idx)
    rs.shuffle(tr_idx); rs.shuffle(ts_idx)
    return X[tr_idx], X[ts_idx], y[tr_idx], y[ts_idx]

class KNNManual:
    def __init__(self, k=7):
        self.k = k
    def fit(self, X, y):
        self.X, self.y = X, y
        return self
    def _dist(self, A):
        a2 = (A ** 2).sum(1)[:, None]
        b2 = (self.X ** 2).sum(1)[None, :]
        return np.sqrt(np.maximum(a2 + b2 - 2 * A @ self.X.T, 1e-12))
    def predict_proba1(self, X):
        D = self._dist(X)
        idx = np.argsort(D, axis=1)[:, :self.k]
        out = np.zeros(len(X))
        for i in range(len(X)):
            w = 1.0 / (D[i, idx[i]] + 1e-8)
            yy = self.y[idx[i]]
            out[i] = w[yy == 1].sum() / w.sum()
        return out
    def predict(self, X):
        return (self.predict_proba1(X) >= 0.5).astype(int)

class GaussianNBManual:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean, self.var, self.prior = {}, {}, {}
        for c in self.classes:
            Xc = X[y == c]
            self.mean[c]  = Xc.mean(0)
            self.var[c]   = Xc.var(0) + 1e-6
            self.prior[c] = len(Xc) / len(X)
        return self
    def _log_post(self, X):
        logs = []
        for c in self.classes:
            ll = (-0.5 * np.log(2 * np.pi * self.var[c])
                  - 0.5 * (X - self.mean[c]) ** 2 / self.var[c]).sum(1)
            logs.append(np.log(self.prior[c]) + ll)
        return np.array(logs).T
    def predict_proba1(self, X):
        L = self._log_post(X)
        L = L - L.max(1, keepdims=True)
        P = np.exp(L); P = P / P.sum(1, keepdims=True)
        return P[:, 1]
    def predict(self, X):
        return self.classes[np.argmax(self._log_post(X), axis=1)]

class _TreeNode:
    __slots__ = ("feat", "thr", "left", "right", "proba")
    def __init__(self):
        self.feat = self.thr = self.left = self.right = None
        self.proba = None

class DecisionTreeManual:

    def __init__(self, max_depth=None, min_samples_split=2,
                 max_features=None, seed=SEED):
        self.max_depth = max_depth if max_depth is not None else 10**9
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.rng = np.random.default_rng(seed)
        self.importances_ = None

    @staticmethod
    def _gini_from_counts(pos, n):
        if n == 0:
            return 0.0
        p = pos / n
        return 1.0 - p * p - (1 - p) ** 2

    def _best_split(self, X, y):
        n, d = X.shape
        total_pos = y.sum()
        parent = self._gini_from_counts(total_pos, n)
        if self.max_features:
            mf = max(1, int(self.max_features))
            feats = self.rng.choice(d, size=min(mf, d), replace=False)
        else:
            feats = range(d)
        best = (0.0, None, None)
        for f in feats:
            col = X[:, f]
            order = np.argsort(col)
            cv, cy = col[order], y[order]
            pos_prefix = np.cumsum(cy)
            for i in range(self.min_samples_split - 1, n - self.min_samples_split):
                if cv[i] == cv[i + 1]:
                    continue
                nl = i + 1
                pl = pos_prefix[i]
                gl = self._gini_from_counts(pl, nl)
                gr = self._gini_from_counts(total_pos - pl, n - nl)
                gain = parent - (nl * gl + (n - nl) * gr) / n
                if gain > best[0]:
                    best = (gain, f, (cv[i] + cv[i + 1]) / 2.0)
        return best

    def _build(self, X, y, depth):
        node = _TreeNode()
        node.proba = y.mean() if len(y) else 0.0
        if (depth >= self.max_depth or len(y) < self.min_samples_split
                or y.min() == y.max()):
            return node
        gain, f, thr = self._best_split(X, y)
        if f is None or gain <= 0:
            return node
        self.importances_[f] += gain * len(y)
        mask = X[:, f] <= thr
        if mask.all() or (~mask).all():
            return node
        node.feat, node.thr = f, thr
        node.left  = self._build(X[mask],  y[mask],  depth + 1)
        node.right = self._build(X[~mask], y[~mask], depth + 1)
        return node

    def fit(self, X, y):
        self.importances_ = np.zeros(X.shape[1])
        self.root = self._build(X, y, 0)
        s = self.importances_.sum()
        if s > 0:
            self.importances_ /= s
        return self

    def _proba_one(self, x):
        node = self.root
        while node.feat is not None:
            node = node.left if x[node.feat] <= node.thr else node.right
        return node.proba

    def predict_proba1(self, X):
        return np.array([self._proba_one(x) for x in X])
    def predict(self, X):
        return (self.predict_proba1(X) >= 0.5).astype(int)

class RandomForestManual:
    def __init__(self, n_estimators=40, max_depth=14, seed=SEED):
        self.n = n_estimators
        self.max_depth = max_depth
        self.seed = seed
    def fit(self, X, y):
        n, d = X.shape
        mf = max(1, int(np.sqrt(d)))
        self.trees = []
        self.importances_ = np.zeros(d)
        rs = np.random.default_rng(self.seed)
        for t in range(self.n):
            idx = rs.integers(0, n, size=n)
            tree = DecisionTreeManual(max_depth=self.max_depth,
                                      max_features=mf, seed=self.seed + t)
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)
            self.importances_ += tree.importances_
        self.importances_ /= self.n
        return self
    def predict_proba1(self, X):
        return np.mean([t.predict_proba1(X) for t in self.trees], axis=0)
    def predict(self, X):
        return (self.predict_proba1(X) >= 0.5).astype(int)

class AdaBoostManual:

    def __init__(self, n_estimators=50):
        self.M = n_estimators
    @staticmethod
    def _fit_stump(X, y_pm, w):
        n, d = X.shape
        best = (np.inf, None, None, 1)
        for f in range(d):
            col = X[:, f]
            order = np.argsort(col)
            cv = col[order]; yo = y_pm[order]; wo = w[order]

            w_pos = np.cumsum(wo * (yo == 1))
            w_neg = np.cumsum(wo * (yo == -1))
            tot_pos = w_pos[-1]; tot_neg = w_neg[-1]
            for i in range(n - 1):
                if cv[i] == cv[i + 1]:
                    continue

                err_p = w_pos[i] + (tot_neg - w_neg[i])
                err_m = w_neg[i] + (tot_pos - w_pos[i])
                if err_p < best[0]:
                    best = (err_p, f, (cv[i]+cv[i+1])/2, +1)
                if err_m < best[0]:
                    best = (err_m, f, (cv[i]+cv[i+1])/2, -1)
        return best
    @staticmethod
    def _stump_pred(X, f, thr, pol):
        p = np.where(X[:, f] <= thr, -1, 1)
        return p if pol == 1 else -p
    def fit(self, X, y):
        n = len(y)
        y_pm = np.where(y == 1, 1, -1)
        w = np.ones(n) / n
        self.stumps, self.alpha = [], []
        for m in range(self.M):
            err, f, thr, pol = self._fit_stump(X, y_pm, w)
            err = min(max(err, 1e-10), 1 - 1e-10)
            alpha = 0.5 * np.log((1 - err) / err)
            pred = self._stump_pred(X, f, thr, pol)
            w = w * np.exp(-alpha * y_pm * pred)
            w /= w.sum()
            self.stumps.append((f, thr, pol))
            self.alpha.append(alpha)
        return self
    def decision(self, X):
        s = np.zeros(len(X))
        for (f, thr, pol), a in zip(self.stumps, self.alpha):
            s += a * self._stump_pred(X, f, thr, pol)
        return s
    def predict_proba1(self, X):
        return 1.0 / (1.0 + np.exp(-2 * self.decision(X)))
    def predict(self, X):
        return (self.decision(X) >= 0).astype(int)

class LinearSVMManual:

    def __init__(self, C=1.0, epochs=30, seed=SEED):
        self.C = C; self.epochs = epochs; self.seed = seed
    def fit(self, X, y):
        n, d = X.shape
        Xb = np.hstack([X, np.ones((n, 1))])
        y_pm = np.where(y == 1, 1.0, -1.0)
        lam = 1.0 / (self.C * n)
        w = np.zeros(d + 1)
        rs = np.random.default_rng(self.seed)
        t = 0
        for _ in range(self.epochs):
            for i in rs.permutation(n):
                t += 1
                eta = 1.0 / (lam * t)
                if y_pm[i] * (Xb[i] @ w) < 1:
                    w = (1 - eta * lam) * w + eta * y_pm[i] * Xb[i]
                else:
                    w = (1 - eta * lam) * w
        self.w = w
        return self
    def decision(self, X):
        return np.hstack([X, np.ones((len(X), 1))]) @ self.w
    def predict_proba1(self, X):
        return 1.0 / (1.0 + np.exp(-self.decision(X)))
    def predict(self, X):
        return (self.decision(X) >= 0).astype(int)

class RBFSVMManual:

    def __init__(self, C=1.0, gamma=None, tol=1e-3, max_passes=5, seed=SEED):
        self.C = C; self.gamma = gamma; self.tol = tol
        self.max_passes = max_passes; self.seed = seed
    def _kernel(self, A, B):
        a2 = (A ** 2).sum(1)[:, None]
        b2 = (B ** 2).sum(1)[None, :]
        d2 = np.maximum(a2 + b2 - 2 * A @ B.T, 0)
        return np.exp(-self.gamma * d2)
    def fit(self, X, y):
        n = len(y)
        if self.gamma is None:
            self.gamma = 1.0 / (X.shape[1] * X.var())
        ypm = np.where(y == 1, 1.0, -1.0)
        K = self._kernel(X, X)
        alpha = np.zeros(n); b = 0.0
        rs = np.random.default_rng(self.seed)
        passes = 0
        while passes < self.max_passes:
            changed = 0
            for i in range(n):
                Ei = (alpha * ypm) @ K[i] + b - ypm[i]
                if ((ypm[i]*Ei < -self.tol and alpha[i] < self.C) or
                    (ypm[i]*Ei >  self.tol and alpha[i] > 0)):
                    j = rs.integers(0, n)
                    while j == i:
                        j = rs.integers(0, n)
                    Ej = (alpha * ypm) @ K[j] + b - ypm[j]
                    ai, aj = alpha[i], alpha[j]
                    if ypm[i] != ypm[j]:
                        L = max(0, aj - ai); Hc = min(self.C, self.C + aj - ai)
                    else:
                        L = max(0, ai + aj - self.C); Hc = min(self.C, ai + aj)
                    if L == Hc:
                        continue
                    eta = 2*K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue
                    alpha[j] = aj - ypm[j]*(Ei - Ej)/eta
                    alpha[j] = min(Hc, max(L, alpha[j]))
                    if abs(alpha[j] - aj) < 1e-5:
                        continue
                    alpha[i] = ai + ypm[i]*ypm[j]*(aj - alpha[j])
                    b1 = b - Ei - ypm[i]*(alpha[i]-ai)*K[i,i] - ypm[j]*(alpha[j]-aj)*K[i,j]
                    b2 = b - Ej - ypm[i]*(alpha[i]-ai)*K[i,j] - ypm[j]*(alpha[j]-aj)*K[j,j]
                    if 0 < alpha[i] < self.C:
                        b = b1
                    elif 0 < alpha[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2
                    changed += 1
            passes = passes + 1 if changed == 0 else 0
        self.X = X; self.ypm = ypm; self.alpha = alpha; self.b = b
        return self
    def decision(self, X):
        return (self.alpha * self.ypm) @ self._kernel(self.X, X) + self.b
    def predict_proba1(self, X):
        return 1.0 / (1.0 + np.exp(-self.decision(X)))
    def predict(self, X):
        return (self.decision(X) >= 0).astype(int)

def confusion_matrix_manual(y, p):
    cm = np.zeros((2, 2), dtype=int)
    for t, pr in zip(y, p):
        cm[t][pr] += 1
    return cm

def precision_recall_f1(cm, cls):
    tp = cm[cls, cls]
    fp = cm[:, cls].sum() - tp
    fn = cm[cls, :].sum() - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec  = tp / (tp + fn) if tp + fn else 0.0
    f1   = 2*prec*rec/(prec+rec) if prec+rec else 0.0
    return prec, rec, f1

def roc_curve_manual(y, score):
    order = np.argsort(-score)
    ys = y[order]
    P = (y == 1).sum(); N = (y == 0).sum()
    tp = np.cumsum(ys == 1); fp = np.cumsum(ys == 0)
    tpr = np.concatenate([[0], tp / P])
    fpr = np.concatenate([[0], fp / N])
    return fpr, tpr

def auc_manual(fpr, tpr):

    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))

def evaluate(clf, X_tr, y_tr, X_ts, y_ts, name, i, total):
    print(f"\n[{i}/{total}] Навчання: {name}...", flush=True)
    t0 = time.time(); clf.fit(X_tr, y_tr); train_t = time.time() - t0
    t1 = time.time(); preds = clf.predict(X_ts)
    inf_ms = (time.time() - t1) / len(y_ts) * 1000
    score = clf.predict_proba1(X_ts)

    cm = confusion_matrix_manual(y_ts, preds)
    acc = (preds == y_ts).mean()
    pf, rf, f1f = precision_recall_f1(cm, 0)
    pr, rr, f1r = precision_recall_f1(cm, 1)
    fpr, tpr = roc_curve_manual(y_ts, score)
    roc_auc = auc_manual(fpr, tpr)

    print(f"  Accuracy={acc*100:.2f}%  Recall(Rotten)={rr*100:.1f}%  "
          f"AUC={roc_auc:.3f}  train={train_t:.2f}s  inf={inf_ms:.3f}ms")
    return {
        "name": name, "accuracy": float(acc),
        "precision_fresh": pf, "recall_fresh": rf, "f1_fresh": f1f,
        "precision_rotten": pr, "recall_rotten": rr, "f1_rotten": f1r,
        "auc": roc_auc, "train_time_s": float(train_t),
        "inference_ms_per_image": float(inf_ms),
        "fpr": fpr.tolist(), "tpr": tpr.tolist(),
        "preds": preds.tolist(), "labels": y_ts.tolist(),
    }

def plot_accuracy_bar(results):
    names = [r["name"] for r in results]
    accs  = [r["accuracy"]*100 for r in results]
    recs  = [r["recall_rotten"]*100 for r in results]
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - w/2, accs, w, label="Accuracy (%)", color="#4c72b0")
    ax.bar(x + w/2, recs, w, label="Recall Rotten (%)", color="#dd8452")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 115); ax.set_ylabel("Відсоток (%)")
    ax.set_title("Accuracy та Recall(Rotten)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig("classical_accuracy_bar.png", dpi=150); plt.close()

def plot_confusion(results):
    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    axes = axes.flat
    for i, res in enumerate(results):
        cm = confusion_matrix_manual(np.array(res["labels"]), np.array(res["preds"]))
        ax = axes[i]
        ax.imshow(cm, cmap="Blues")
        for a in range(2):
            for b in range(2):
                ax.text(b, a, str(cm[a, b]), ha="center", va="center",
                        fontsize=16, fontweight="bold",
                        color="white" if cm[a, b] > cm.max()*0.6 else "#1f2d3d")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
        ax.set_title(f"{res['name']}\nAcc={res['accuracy']:.3f}", fontsize=10)
        ax.set_xlabel("Передбачений"); ax.set_ylabel("Справжній")
    for j in range(len(results), 9):
        axes[j].axis("off")
    plt.suptitle("Матриці плутанини (7 алгоритмів)", fontsize=13)
    plt.tight_layout()
    plt.savefig("classical_confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_roc(results):
    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    for i, r in enumerate(results):
        plt.plot(r["fpr"], r["tpr"], lw=2, color=cmap(i),
                 label=f"{r['name']}  AUC={r['auc']:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC-криві"); plt.legend(fontsize=9, loc="lower right")
    plt.tight_layout(); plt.savefig("classical_roc_curves.png", dpi=150); plt.close()

def plot_feature_importance(imp, hog_n, lbp_n):
    groups = {
        f"HOG ({hog_n})":  imp[:hog_n].sum(),
        f"LBP ({lbp_n})":  imp[hog_n:hog_n+lbp_n].sum(),
        f"Color ({len(imp)-hog_n-lbp_n})": imp[hog_n+lbp_n:].sum(),
    }
    plt.figure(figsize=(6, 4))
    bars = plt.bar(groups.keys(), groups.values(),
                   color=["#4daf4a", "#377eb8", "#e41a1c"])
    for b, v in zip(bars, groups.values()):
        plt.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
                 f"{v:.3f}", ha="center")
    plt.ylabel("Сумарна важливість")
    plt.title("Внесок груп ознак (Random Forest)")
    plt.tight_layout(); plt.savefig("feature_importance.png", dpi=150); plt.close()

def main():
    paths, labels = collect_samples(DATA_ROOT)

    t = time.time()
    X, y = build_feature_matrix(paths, labels)
    feat_time = time.time() - t
    print(f"Час вилучення ознак: {feat_time:.1f}s")

    X_tr_raw, X_ts_raw, y_tr, y_ts = train_test_split_manual(X, y, 0.30)
    print(f"Спліт: train={len(y_tr)}, test={len(y_ts)}")

    scaler = StandardScalerManual()
    X_tr_sc = scaler.fit_transform(X_tr_raw)
    X_ts_sc = scaler.transform(X_ts_raw)

    pca = PCAManual(var_keep=0.97)
    X_tr = pca.fit_transform(X_tr_sc)
    X_ts = pca.transform(X_ts_sc)
    print(f"PCA: {X.shape[1]} -> {pca.n_components_} компонент")

    classifiers = [
        ("KNN",           KNNManual(k=7)),
        ("Linear SVM",    LinearSVMManual(C=1.0, epochs=30)),
        ("RBF SVM",       RBFSVMManual(C=1.0)),
        ("Decision Tree", DecisionTreeManual(max_depth=None)),
        ("Random Forest", RandomForestManual(n_estimators=40, max_depth=14)),
        ("AdaBoost",      AdaBoostManual(n_estimators=50)),
        ("Naive Bayes",   GaussianNBManual()),
    ]
    results = []
    for i, (name, clf) in enumerate(classifiers, 1):
        results.append(evaluate(clf, X_tr, y_tr, X_ts, y_ts,
                                name, i, len(classifiers)))

    print("\n" + "=" * 70)
    print(f"{'Метод':<16}{'Acc':>8}{'Rec(Rot)':>10}{'AUC':>8}{'Inf,ms':>9}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<16}{r['accuracy']*100:>7.1f}%"
              f"{r['recall_rotten']*100:>9.1f}%{r['auc']:>8.3f}"
              f"{r['inference_ms_per_image']:>9.3f}")
    best = max(results, key=lambda r: r["accuracy"])
    print(f"\nНайкращий: {best['name']} ({best['accuracy']*100:.1f}%)")

    print("\nОбчислення важливості ознак (Random Forest на 1840 ознаках)...")
    hog_n = len(extract_hog_manual(np.zeros((IMG_SIZE, IMG_SIZE), np.uint8)))
    lbp_n = 26
    rf_imp = RandomForestManual(n_estimators=20, max_depth=12)
    rf_imp.fit(X_tr_sc, y_tr)

    print("Генерація графіків...")
    plot_accuracy_bar(results)
    plot_confusion(results)
    plot_roc(results)
    plot_feature_importance(rf_imp.importances_, hog_n, lbp_n)

    output = {
        "feature_extraction_time_s": float(feat_time),
        "feature_dim_original": int(X.shape[1]),
        "feature_dim_pca": int(pca.n_components_),
        "train_images": int(len(y_tr)),
        "test_images": int(len(y_ts)),
        "classifiers": results,
    }
    with open("results_classical.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print("\nЗбережено: results_classical.json, classical_*.png, feature_importance.png")

if __name__ == "__main__":
    main()

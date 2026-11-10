"""Baseline checkpoint: outlet-only vs ridge on content, month-blocked nested CV.

Run:  python baseline.py path/to/labeled_data_10_publishers_2024_v1.csv
Needs: pandas numpy scipy scikit-learn matplotlib joblib
Writes: results.txt (paste this back) and baseline.png
"""
import os, pickle, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED, K, B = 0, 5, 1000
ALPHAS = [0.1, 0.3, 1, 3, 10, 30, 100]
TONE = ["sent_tone_neg", "sent_tone_neu", "sent_tone_pos"]
TYPE = ["sent_type_fac", "sent_type_opn", "sent_type_bor", "sent_type_quo", "sent_type_oth"]
FOCUS = ["sent_focus_dem", "sent_focus_rep", "sent_focus_bth", "sent_focus_non"]
CATS = ["topic", "subtopic", "news_type"]
out = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)

# ---------- load ----------
path = sys.argv[1] if len(sys.argv) > 1 else "labeled_data_10_publishers_2024_v1.csv"
allcols = list(pd.read_csv(path, nrows=0).columns)
TEXT = "takeaways" if "takeaways" in allcols else "summary"
USE = ["publisher_full", "datetime", TEXT, "category", "article_lean"] + CATS + TONE + TYPE + FOCUS
raw = pd.read_csv(path, usecols=USE, low_memory=False)   # rationale columns and other GPT scores never loaded
say(f"loaded rows={len(raw)} cols={len(allcols)} text_col={TEXT}")
df = raw[raw["category"] == "Politics"].copy(); del raw
n_pol = len(df)
df = df.dropna(subset=["article_lean", TEXT, "datetime", "publisher_full"])
df = df[df[TEXT].astype(str).str.strip() != ""]
say(f"politics rows={n_pol} dropped_missing={n_pol - len(df)} used={len(df)}")
dt = pd.to_datetime(df["datetime"], utc=True, errors="coerce", format="ISO8601").dt.tz_convert("America/New_York")
say(f"timestamps without timezone (treated as UTC)={(~df['datetime'].astype(str).str.contains(r'[+-]\d\d:\d\d$')).sum()}")
assert dt.notna().all(), "unparseable datetimes"
df["month"], df["day"] = dt.dt.month.values, dt.dt.strftime("%Y-%m-%d").values
df = df.reset_index(drop=True)
y = df["article_lean"].astype(float).values
pub = df["publisher_full"].astype(str).values
for c in CATS: df[c] = df[c].fillna("NA").astype(str)

def shares(cols):
    a = df[cols].fillna(0).astype(float).values
    t = a.sum(1, keepdims=True)
    return np.divide(a, t, out=np.zeros_like(a), where=t > 0)
S = np.hstack([shares(TONE), shares(TYPE), shares(FOCUS)])   # row-wise, no leakage
say(f"rows with zero sentence counts={(df[TONE].fillna(0).sum(axis=1) == 0).sum()}")
say(f"summary median words={int(df[TEXT].astype(str).str.split().str.len().median())}")

means = pd.Series(y).groupby(pub).mean().sort_values()
order = list(means.index)
def grp(p):
    q = p.lower()
    return "right" if ("breitbart" in q or "fox" in q) else "wsj" if "wall street" in q else "left"
group = np.array([grp(p) for p in pub])
say("outlet mean lean, n, group:")
for p in order: say(f"  {p}: {means[p]:+.2f} n={(pub == p).sum()} {grp(p)}")

# ---------- month-blocked folds ----------
months = np.sort(df["month"].unique())
perm = np.random.RandomState(SEED).permutation(months)
fold_of_month = {m: i % K for i, m in enumerate(perm)}
fold = df["month"].map(fold_of_month).values
say("fold -> months:", {k: sorted(int(m) for m in months if fold_of_month[m] == k) for k in range(K)})

def outlet_means(tr):
    m = pd.Series(y[tr]).groupby(pub[tr]).mean()
    return pd.Series(pub).map(m).fillna(y[tr].mean()).values   # for all rows

def design(tr, te):
    """All preprocessing fit on tr only. Returns (Xtr, Xte) without and with sentence shares."""
    tf = TfidfVectorizer(lowercase=True, min_df=5, sublinear_tf=True)
    oh = OneHotEncoder(handle_unknown="ignore")
    sc = StandardScaler()
    A = sp.hstack([tf.fit_transform(df[TEXT].values[tr]), oh.fit_transform(df[CATS].values[tr])]).tocsr()
    Bm = sp.hstack([tf.transform(df[TEXT].values[te]), oh.transform(df[CATS].values[te])]).tocsr()
    Str, Ste = sc.fit_transform(S[tr]), sc.transform(S[te])
    return {"text+topic": (A, Bm),
            "text+topic+counts": (sp.hstack([A, sp.csr_matrix(Str)]).tocsr(), sp.hstack([Bm, sp.csr_matrix(Ste)]).tocsr())}

def fit_predict(tr, te, alphas):
    """Ridge on two targets: lean (content-only) and lean minus outlet training mean (residual)."""
    om = outlet_means(tr)
    Y = np.column_stack([y[tr], y[tr] - om[tr]])
    res = {}
    for name, (Xtr, Xte) in design(tr, te).items():
        for a in alphas:
            res[(name, a)] = Ridge(alpha=a, solver="sparse_cg", tol=1e-4, random_state=SEED).fit(Xtr, Y).predict(Xte)
    return om[te], res

def outer(k):
    cache = f"fold{k}_seed{SEED}.pkl"          # delete these files to recompute from scratch
    if os.path.exists(cache): return pickle.load(open(cache, "rb"))
    r = _outer(k); pickle.dump(r, open(cache, "wb")); return r

def _outer(k):
    te = np.where(fold == k)[0]; tr = np.where(fold != k)[0]
    # inner month-blocked CV over the 4 training folds
    sse = {}
    for j in range(K):
        if j == k: continue
        itr = np.where((fold != k) & (fold != j))[0]; ite = np.where(fold == j)[0]
        om, res = fit_predict(itr, ite, ALPHAS)
        for (name, a), P in res.items():
            e = np.array([((y[ite] - P[:, 0]) ** 2).sum(), ((y[ite] - om - P[:, 1]) ** 2).sum()])
            sse[(name, a)] = sse.get((name, a), 0) + e
    best = {(name, t): min(ALPHAS, key=lambda a: sse[(name, a)][t]) for name in ["text+topic", "text+topic+counts"] for t in (0, 1)}
    om, res = fit_predict(tr, te, sorted(set(best.values())))
    preds = {(name, t): res[(name, a)][:, t] for (name, t), a in best.items()}
    return k, te, om, preds, best

t0 = time.time()
runs = Parallel(n_jobs=int(os.environ.get('N_JOBS', 1)))(delayed(outer)(k) for k in range(K))
say(f"nested CV done in {time.time() - t0:.0f}s")

n = len(y)
p_out = np.zeros(n); P = {}
for k, te, om, preds, best in sorted(runs, key=lambda r: r[0]):
    p_out[te] = om
    for key, v in preds.items(): P.setdefault(key, np.zeros(n))[te] = v
    say(f"fold {k} chosen alpha:", {f"{a}|{'lean' if t == 0 else 'resid'}": al for (a, t), al in best.items()})
say("alpha grid:", ALPHAS)

# ---------- scores with day-bootstrap intervals ----------
days, d_idx = np.unique(df["day"].values, return_inverse=True)
W = np.random.RandomState(SEED).multinomial(len(days), np.ones(len(days)) / len(days), size=B).astype(float)
say(f"news days={len(days)} bootstrap resamples={B}")
def daysum(v, mask): return np.bincount(d_idx[mask], weights=v[mask], minlength=len(days))

def r2_overall(pred, mask):          # 1 - SSE / SST, SST around the (resampled) mean of y
    c, s1, s2, se = (daysum(v, mask) for v in (np.ones(n), y, y * y, (y - pred) ** 2))
    f = lambda w: 1 - (w @ se) / (w @ s2 - (w @ s1) ** 2 / (w @ c))
    return f(np.ones(len(days))), np.array([f(w) for w in W])
def r2_within(pred, mask):           # 1 - SSE / SSE(outlet-only); 0 = no gain over outlet
    se, s0 = daysum((y - pred) ** 2, mask), daysum((y - p_out) ** 2, mask)
    f = lambda w: 1 - (w @ se) / (w @ s0)
    return f(np.ones(len(days))), np.array([f(w) for w in W])
def fmt(pt_bs):
    pt, bs = pt_bs; lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return f"{pt:.3f} [{lo:.3f}, {hi:.3f}]", pt, lo, hi

allm = np.ones(n, bool)
masks = {"all": allm, "right": group == "right", "left": group == "left", "wsj": group == "wsj"}
say("\nRESULTS (point [95% day-bootstrap interval])")
say("outlet-only overall CV R2:", fmt(r2_overall(p_out, allm))[0])
for name in ["text+topic+counts", "text+topic"]:
    say(f"ridge {name}:")
    say("  content-only overall R2 (no outlet info):", fmt(r2_overall(P[(name, 0)], allm))[0])
    comb = p_out + P[(name, 1)]
    say("  outlet mean + ridge on residual, overall R2:", fmt(r2_overall(comb, allm))[0])
    pt, bs = r2_overall(comb, allm); pt0, bs0 = r2_overall(p_out, allm)
    lo, hi = np.percentile(bs - bs0, [2.5, 97.5])
    say(f"  gain over outlet-only: {pt - pt0:.3f} [{lo:.3f}, {hi:.3f}]")
    for g, m in masks.items():
        say(f"  within-outlet R2 {g} (n={m.sum()}):", fmt(r2_within(comb, m))[0])
pr, br = r2_within(p_out + P[("text+topic+counts", 1)], masks["right"])
pl, bl = r2_within(p_out + P[("text+topic+counts", 1)], masks["left"])
lo, hi = np.nanpercentile(bl - br, [2.5, 97.5])
say(f"left minus right within-outlet R2 (with counts): {pl - pr:.3f} [{lo:.3f}, {hi:.3f}]")

# ---------- figure ----------
fig, ax = plt.subplots(figsize=(10, 4.8))
x = np.arange(len(order)); w = 0.38
say("\nper-outlet within-outlet R2:")
for i, (name, col) in enumerate([("text+topic+counts", "#3b6ea5"), ("text+topic", "#a9bcd0")]):
    vals = [fmt(r2_within(p_out + P[(name, 1)], pub == p)) for p in order]
    pts = np.array([v[1] for v in vals]); lo = np.array([v[2] for v in vals]); hi = np.array([v[3] for v in vals])
    ax.bar(x + (i - 0.5) * w, pts, w, color=col, label=f"ridge: {name}", yerr=[pts - lo, hi - pts], capsize=3, ecolor="#333")
    for p, v in zip(order, vals): say(f"  {name} | {p}: {v[0]}")
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels([f"{p}\n({means[p]:+.2f})" for p in order], rotation=30, ha="right", fontsize=8)
ax.set_xlabel("Outlet, ordered by mean lean (mean in parentheses)")
ax.set_ylabel("Out-of-fold within-outlet R$^2$")
ax.set_title("Ridge on content vs outlet-only (0 = no gain); 95% day-bootstrap intervals", fontsize=10)
ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig("baseline.png", dpi=200)
open("results.txt", "w").write("\n".join(out) + "\n")
print("\nwrote results.txt and baseline.png")

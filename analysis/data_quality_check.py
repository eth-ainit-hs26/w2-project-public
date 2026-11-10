import ast, sys, json
import numpy as np, pandas as pd

ROOT = sys.argv[1]
cust = pd.read_csv(f"{ROOT}/data/customers.tsv", sep="\t")
prod = pd.read_csv(f"{ROOT}/data/products.tsv", sep="\t")
C = np.array([ast.literal_eval(v) for v in cust.FeatureVector], float)
I = np.array([ast.literal_eval(v) for v in prod.FeatureVector], float)
price, cost = prod.Price.to_numpy(float), prod.Cost.to_numpy(float)
margin = price - cost
K, ITEMS_PER_CUSTOMER = 5, 15


def kappa_cos(Cv, Iv):
    cn = Cv / np.linalg.norm(Cv, axis=1, keepdims=True)
    inn = Iv / np.linalg.norm(Iv, axis=1, keepdims=True)
    return np.clip((cn @ inn.T + 1) / 2, 0, 1)


def report(name, kap, margin=margin):
    out = {"name": name}
    out["kappa_mean"] = kap.mean()
    out["kappa_p10_p90"] = np.percentile(kap, [10, 90]).round(3).tolist()
    out["share_kappa_gt_0.5"] = (kap > 0.5).mean()
    # two-way decomposition of kappa variance: item main effect / customer main effect / interaction
    g = kap.mean()
    ci = kap.mean(1, keepdims=True) - g
    it = kap.mean(0, keepdims=True) - g
    inter = kap - g - ci - it
    tot = ((kap - g) ** 2).sum()
    out["var_item"] = (it ** 2).sum() * kap.shape[0] / tot
    out["var_cust"] = (ci ** 2).sum() * kap.shape[1] / tot
    out["var_interaction"] = (inter ** 2).sum() / tot
    # expected revenue per day for each policy (exact expectations, no sampling)
    ev = kap * margin  # expected margin per (c,i)
    oracle = np.sort(ev, 1)[:, -K:].sum()
    top_margin = np.argsort(margin)[-K:]
    highest_margin = ev[:, top_margin].sum()
    # best single non-personalized list (same K items for everyone)
    best_global = np.sort(ev.sum(0))[-K:].sum()
    random_ = ev.mean(1).sum() * K
    out.update(oracle=oracle, highest_margin=highest_margin, best_global_list=best_global, random=random_)
    out["headroom_oracle_vs_highest_margin_%"] = 100 * (oracle / highest_margin - 1)
    out["headroom_oracle_vs_bestglobal_%"] = 100 * (oracle / best_global - 1)
    # personalization: how many distinct oracle top-K sets / overlap with highest-margin list
    tops = np.argsort(ev, 1)[:, -K:]
    out["distinct_oracle_sets"] = len({tuple(sorted(t)) for t in tops})
    out["avg_overlap_with_highest_margin_list"] = np.mean([len(set(t) & set(top_margin)) for t in tops])
    return out


res = []
kap5 = kappa_cos(C, I)
res.append(report("current (5-D TSV, cosine)", kap5))

# --- variant: the "Project 2 patch" from data_generation.ipynb (never actually loaded) ---
rng = np.random.default_rng(42)
mult = rng.beta(a=6, b=1.2, size=5 * 40).reshape(40, 5)
C6 = np.hstack((C * mult, np.ones((40, 1))))
pf = 2 * (1 / (1 + np.exp(0.02 * (price - 180)))) - 1
I6 = np.hstack((I, pf[:, None]))
res.append(report("patch from data_generation.ipynb (6-D, price feature)", kappa_cos(C6, I6)))

# --- variant: sharper kernel, cosine -> sigmoid with low base rate ---
cos = 2 * kap5 - 1
for a, b in [(4, -2), (6, -3)]:
    res.append(report(f"sigmoid({a}*cos{b:+d})", 1 / (1 + np.exp(-(a * cos + b)))))

# --- variant: mean-centred features (remove the shared 'everyone likes everything' direction) ---
Cc, Ic = C - C.mean(0), I - I.mean(0)
res.append(report("mean-centred features, cosine", kappa_cos(Cc, Ic)))
res.append(report("mean-centred + sigmoid(6*cos-3)", 1 / (1 + np.exp(-(6 * (2 * kappa_cos(Cc, Ic) - 1) - 3)))))

# --- why training revenue is flat: exposure set is fixed, so E[revenue] is constant ---
np.random.seed(42)
item_ids = list(range(len(prod)))
train_sets = [np.random.choice(item_ids, size=ITEMS_PER_CUSTOMER, replace=False) for _ in range(len(cust))]
mu = sum((kap5[c, s] * margin[s]).sum() for c, s in enumerate(train_sets))
sd = np.sqrt(sum((kap5[c, s] * (1 - kap5[c, s]) * margin[s] ** 2).sum() for c, s in enumerate(train_sets)))
train = {"E_train_revenue": mu, "sd_per_iteration": sd}

# --- simulate what the training curve looks like if the agent picks top-K inside its training set ---
def sim(kap, steer, iters=300, d=4, lr=0.01, reg=1e-4, wp=0.5, wn=0.5, seed=0):
    r = np.random.default_rng(seed)
    U = r.uniform(0, 1, (len(cust), d)); V = r.uniform(0, 1, (len(prod), d))
    bc = np.zeros(len(cust)); bi = np.zeros(len(prod))
    sig = lambda z: 1 / (1 + np.exp(-z))
    exp_rev = []
    for _ in range(iters):
        er = 0.0
        for c, s in enumerate(train_sets):
            if steer:
                p = sig(U[c] @ V[s].T + bc[c] + bi[s])
                shown = s[np.argsort(p * margin[s])[-K:]]
            else:
                shown = r.choice(s, K, replace=False)
            er += (kap[c, shown] * margin[shown]).sum()
            bought = r.random(K) < kap[c, shown]
            for i, y in zip(shown, bought):
                p = sig(U[c] @ V[i] + bc[c] + bi[i]); e = float(y) - p; w = wp if y else wn
                U[c] += lr * (w * e * V[i] - reg * U[c]); V[i] += lr * (w * e * U[c] - reg * V[i])
                bc[c] += lr * (w * e - reg * bc[c]); bi[i] += lr * (w * e - reg * bi[i])
        exp_rev.append(er)
    return np.array(exp_rev)

curves = {k: sim(kap5, k == "steered") for k in ["random-in-set", "steered"]}
train["curves_first_last_10_mean"] = {k: [v[:10].mean(), v[-10:].mean()] for k, v in curves.items()}
oracle_in_set = sum(np.sort(kap5[c, s] * margin[s])[-K:].sum() for c, s in enumerate(train_sets))
train["oracle_top5_in_training_set"] = oracle_in_set
np.save(sys.argv[2] + "/curves.npy", np.vstack([curves["random-in-set"], curves["steered"]]))

# --- catalogue sanity ---
cat = {
    "price_range": [price.min(), price.max()], "margin_range": [margin.min(), margin.max()],
    "margin_pct_range": [(margin / price).min().round(3), (margin / price).max().round(3)],
    "top5_margin_items": prod.loc[np.argsort(margin)[-5:], ["Name", "Price", "Cost"]].to_dict("records"),
    "negative_or_zero_margin": int((margin <= 0).sum()),
    "dup_customer_vectors": int(len(C) - len({tuple(v) for v in C})),
    "dup_item_vectors": int(len(I) - len({tuple(v) for v in I})),
    "customer_vec_mean": C.mean(0).round(2).tolist(), "item_vec_mean": I.mean(0).round(2).tolist(),
    "segments": cust.Segment.value_counts().to_dict(),
}
print(json.dumps({"variants": res, "training": train, "catalogue": cat}, indent=1, default=float))

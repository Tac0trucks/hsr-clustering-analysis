#!/usr/bin/env python3
"""
Archetype Mining: Classifying Role Transgression in Character Base Stats
using Honkai: Star Rail.

Pipeline
--------
1. Ingest + sanitize  : merge characters.csv + character_stats.csv, derive Lv.80 stats
2. Preprocess         : features (HP, ATK, DEF, SPD [+ optional Max Energy]) -> StandardScaler
3. K-Means            : scan k in [3, 8] (inertia + silhouette), fit final model, centroids
4. DBSCAN             : k-distance eps heuristic, flag noise (-1) as outliers
5. Visuals            : PCA/t-SNE, Path x Cluster heatmap, centroid radar
6. Analysis tables    : transgressors, anomaly audit, alignment metrics

Usage
-----
    python archetype_mining.py --data-dir /path/to/csvs --out-dir outputs
    python archetype_mining.py --k 5 --eps 0.9 --min-samples 4
    python archetype_mining.py --include-energy      # sensitivity run (see note in README block)

Data note
---------
The Kaggle dump is split across several CSVs (there is no single hsr_characters.csv):
  characters.csv       -> id, name, rarity, path, element, max_energy, base_spd, ...
  character_stats.csv  -> per-ascension base + per-level add for ATK / DEF / HP
Level-80 stat = value_at_ascension_6 + per_level_add * (80 - 1).
SPD has no level scaling in HSR, so `base_spd` is already the Lv.80 value.
`max_energy` is NOT reliably an ultimate cost in this dump (e.g. Acheron = 9, Evanescia = 480,
Castorice = NaN, because stack/point-based ultimates are stored differently). It is therefore
EXCLUDED by default and only used when --include-energy is passed (median-imputed, log1p).
"""
from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (adjusted_rand_score, davies_bouldin_score,
                             normalized_mutual_info_score, silhouette_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:  # optional, only improves label placement
    from adjustText import adjust_text
except ImportError:  # pragma: no cover
    adjust_text = None

warnings.filterwarnings("ignore", category=FutureWarning)
log = logging.getLogger("archetype")

LEVEL = 80
BASE_FEATURES = ["HP", "ATK", "DEF", "SPD"]
SEED = 42


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    data_dir: Path = Path("/mnt/user-data/uploads")
    out_dir: Path = Path("/mnt/user-data/outputs/archetype_results")
    k_min: int = 3
    k_max: int = 8
    k_preferred: tuple = (4, 6)          # silhouette-best k is picked inside this window
    k_override: Optional[int] = None
    eps: Optional[float] = None          # None -> k-distance knee heuristic
    min_samples: int = 4                 # ~ n_features for tiny tabular data
    include_energy: bool = False
    seed: int = SEED
    features: list = field(default_factory=lambda: list(BASE_FEATURES))


# --------------------------------------------------------------------------- #
# 1. Ingestion & preprocessing
# --------------------------------------------------------------------------- #
def load_raw(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    chars = pd.read_csv(cfg.data_dir / "characters.csv")
    stats = pd.read_csv(cfg.data_dir / "character_stats.csv")
    return chars, stats


def derive_level_stats(stats: pd.DataFrame, level: int = LEVEL) -> pd.DataFrame:
    """Lv.`level` stat = (stat at max ascension) + add * (level - 1)."""
    top = stats.loc[stats.groupby("character_id")["ascension"].idxmax()].copy()
    if not (top["ascension"] == 6).all():
        log.warning("Some characters lack ascension 6 rows; using their max ascension.")
    out = pd.DataFrame({"character_id": top["character_id"].values})
    for raw, name in [("hp", "HP"), ("atk", "ATK"), ("def", "DEF")]:
        out[name] = (top[f"{raw}_base"] + top[f"{raw}_add"] * (level - 1)).values
    return out


def make_labels(meta: pd.DataFrame) -> pd.Series:
    """Unique display labels (Trailblazer / March 7th have several variants)."""
    dup = meta["Character Name"].duplicated(keep=False)
    return pd.Series(
        np.where(dup, meta["Character Name"] + " (" + meta["Path"] + ")", meta["Character Name"]),
        index=meta.index,
    )


def build_dataset(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (meta, X_raw) sharing the same index."""
    chars, stats = load_raw(cfg)

    # --- sanitation -------------------------------------------------------- #
    for df, name in [(chars, "characters"), (stats, "stats")]:
        n = df.duplicated().sum()
        if n:
            log.warning("%s: dropping %d exact duplicate rows", name, n)
            df.drop_duplicates(inplace=True)
    assert chars["character_id"].is_unique, "character_id must be unique"

    lvl = derive_level_stats(stats)
    df = chars.merge(lvl, on="character_id", how="inner", validate="1:1")
    if len(df) != len(chars):
        log.warning("%d characters had no stats and were dropped", len(chars) - len(df))

    df = df.rename(columns={"character_name": "Character Name", "path": "Path",
                            "element": "Element", "rarity": "Rarity",
                            "base_spd": "SPD", "max_energy": "Max Energy"})
    for c in BASE_FEATURES:
        if df[c].isna().any():
            raise ValueError(f"Missing values in core feature {c}")
        if (df[c] <= 0).any():
            raise ValueError(f"Non-positive values in {c}")

    # --- metadata isolation ------------------------------------------------ #
    meta = df[["character_id", "Character Name", "Path", "Element", "Rarity"]].copy()
    meta["Label"] = make_labels(meta)
    meta = meta.reset_index(drop=True)

    feats = list(cfg.features)
    X = df[BASE_FEATURES].reset_index(drop=True).copy()
    if cfg.include_energy:
        e = df["Max Energy"].reset_index(drop=True)
        log.warning("Including Max Energy: %d NaN imputed with median; log1p applied.", e.isna().sum())
        X["Max Energy"] = np.log1p(e.fillna(e.median()))
        feats.append("Max Energy")
    cfg.features = feats
    return meta, X[feats]


def scale_features(X: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler


# --------------------------------------------------------------------------- #
# 2A. K-Means
# --------------------------------------------------------------------------- #
def scan_k(Xs: np.ndarray, cfg: Config) -> pd.DataFrame:
    rows = []
    for k in range(cfg.k_min, cfg.k_max + 1):
        km = KMeans(n_clusters=k, n_init=50, random_state=cfg.seed).fit(Xs)
        rows.append({"k": k, "inertia": km.inertia_,
                     "silhouette": silhouette_score(Xs, km.labels_),
                     "davies_bouldin": davies_bouldin_score(Xs, km.labels_),
                     "min_cluster_size": int(np.bincount(km.labels_).min())})
    return pd.DataFrame(rows)


def select_k(scan: pd.DataFrame, cfg: Config) -> int:
    if cfg.k_override:
        return cfg.k_override
    lo, hi = cfg.k_preferred
    window = scan[(scan.k >= lo) & (scan.k <= hi) & (scan.min_cluster_size >= 3)]
    window = window if len(window) else scan
    return int(window.loc[window["silhouette"].idxmax(), "k"])


def fit_kmeans(Xs: np.ndarray, k: int, cfg: Config) -> KMeans:
    return KMeans(n_clusters=k, n_init=100, random_state=cfg.seed).fit(Xs)


def centroid_tables(km: KMeans, scaler: StandardScaler, features: list[str],
                    labels: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.Index(range(km.n_clusters), name="Cluster")
    scaled = pd.DataFrame(km.cluster_centers_, columns=features, index=idx)
    raw = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=features, index=idx)
    raw["n"] = np.bincount(labels, minlength=km.n_clusters)
    return scaled, raw


# --------------------------------------------------------------------------- #
# 2B. DBSCAN
# --------------------------------------------------------------------------- #
def k_distances(Xs: np.ndarray, k: int) -> np.ndarray:
    """Sorted distance of each point to its k-th neighbour (self excluded)."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Xs)
    d, _ = nn.kneighbors(Xs)
    return np.sort(d[:, -1])


def knee_eps(kd: np.ndarray) -> float:
    """Knee = point of max perpendicular distance from the chord of the sorted k-dist curve."""
    n = len(kd)
    x = np.linspace(0, 1, n)
    y = (kd - kd.min()) / (kd.max() - kd.min() + 1e-12)
    dist = np.abs(y - x) / np.sqrt(2)           # chord is y = x after normalising
    return float(kd[int(np.argmax(dist))])


def fit_dbscan(Xs: np.ndarray, cfg: Config) -> tuple[DBSCAN, float, np.ndarray]:
    kd = k_distances(Xs, cfg.min_samples)
    eps = cfg.eps if cfg.eps else knee_eps(kd)
    db = DBSCAN(eps=eps, min_samples=cfg.min_samples).fit(Xs)
    return db, eps, kd


def dbscan_sweep(Xs: np.ndarray, min_samples: int, eps_grid: np.ndarray) -> pd.DataFrame:
    rows = []
    for e in eps_grid:
        lab = DBSCAN(eps=e, min_samples=min_samples).fit_predict(Xs)
        rows.append({"eps": round(float(e), 3), "clusters": len(set(lab) - {-1}),
                     "noise": int((lab == -1).sum())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 3. Analysis helpers
# --------------------------------------------------------------------------- #
def path_alignment(meta: pd.DataFrame, labels: np.ndarray) -> dict:
    ct = pd.crosstab(meta["Path"], labels)
    exp = np.outer(ct.sum(axis=1), ct.sum(axis=0)) / ct.values.sum()
    chi2 = float(((ct.values - exp) ** 2 / exp).sum())
    cramers_v = float(np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1))))
    purity = float(ct.max(axis=1).sum() / ct.values.sum())            # path-centric
    return {"ARI_vs_Path": adjusted_rand_score(meta["Path"], labels),
            "NMI_vs_Path": normalized_mutual_info_score(meta["Path"], labels),
            "ARI_vs_Rarity": adjusted_rand_score(meta["Rarity"], labels),
            "NMI_vs_Rarity": normalized_mutual_info_score(meta["Rarity"], labels),
            "CramersV_Path_Cluster": cramers_v,
            "Path_modal_cluster_share": purity}


def find_transgressors(meta: pd.DataFrame, Xs: np.ndarray, labels: np.ndarray,
                       min_share: float = 0.40) -> pd.DataFrame:
    """
    A character is a *role transgressor* when its K-Means cluster is outside the modal
    cluster(s) of its own Path (ties allowed; Paths whose best cluster holds < 40% of members
    are 'diffuse' and produce no transgressors, since they have no stat-defined home). `fit_margin` adds a continuous view: distance to own-Path
    centroid minus distance to the nearest other-Path centroid (>0 => stats look more like
    another Path). Own-path centroid excludes the character itself (leave-one-out).
    """
    d = meta.copy()
    d["Cluster"] = labels
    def _modal(s: pd.Series):
        vc = s.value_counts()
        top = vc[vc == vc.max()].index.tolist()
        return top  # list; >1 element => tie

    modal_sets = d.groupby("Path")["Cluster"].agg(_modal)
    share = d.groupby("Path")["Cluster"].agg(lambda s: s.value_counts(normalize=True).max())
    d["Path_modal_cluster"] = d["Path"].map(lambda p: "/".join(map(str, modal_sets[p])))
    d["Path_modal_share"] = d["Path"].map(share).round(2)
    # Transgressor = outside ALL of its Path's modal cluster(s), and only if the Path actually
    # has a coherent home (modal share >= min_share). Otherwise the Path is 'diffuse'.
    in_modal = np.array([c in modal_sets[p] for c, p in zip(d["Cluster"], d["Path"])])
    d["Diffuse_Path"] = d["Path_modal_share"] < min_share
    d["Transgressor"] = (~in_modal) & (~d["Diffuse_Path"])

    paths = d["Path"].values
    uniq = sorted(set(paths))
    margins, implied = [], []
    for i in range(len(d)):
        dist = {}
        for p in uniq:
            mask = (paths == p) & (np.arange(len(d)) != i)
            dist[p] = np.linalg.norm(Xs[i] - Xs[mask].mean(0)) if mask.any() else np.inf
        own = dist[paths[i]]
        other = min((v, p) for p, v in dist.items() if p != paths[i])
        margins.append(own - other[0])
        implied.append(min(dist, key=dist.get))
    d["fit_margin"] = np.round(margins, 2)
    d["Stat_implied_Path"] = implied
    return d


def audit_noise(meta: pd.DataFrame, X: pd.DataFrame, Xs: np.ndarray, km: KMeans,
                db_labels: np.ndarray, k: int) -> pd.DataFrame:
    z = pd.DataFrame(Xs, columns=X.columns)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Xs)
    kd = nn.kneighbors(Xs)[0][:, -1]
    rows = []
    for i in np.where(db_labels == -1)[0]:
        zi = z.iloc[i]
        drivers = zi[zi.abs() >= 1.5].sort_values(key=np.abs, ascending=False)
        if drivers.empty:                                    # fall back to the single largest
            drivers = zi.reindex(zi.abs().sort_values(ascending=False).index[:1])
        reason = ", ".join(f"{f} {v:+.1f}σ" for f, v in drivers.items())
        rows.append({"Label": meta.loc[i, "Label"], "Path": meta.loc[i, "Path"],
                     "Rarity": meta.loc[i, "Rarity"], "KMeans_Cluster": int(km.labels_[i]),
                     **{c: round(float(X.iloc[i][c]), 1) for c in X.columns},
                     **{f"z_{c}": round(float(zi[c]), 2) for c in X.columns},
                     "max_|z|": round(float(zi.abs().max()), 2),
                     "isolation_ratio": round(float(kd[i] / np.median(kd)), 2),
                     "flag_reason": reason})
    return pd.DataFrame(rows).sort_values("max_|z|", ascending=False)


# --------------------------------------------------------------------------- #
# 4. Plotting
# --------------------------------------------------------------------------- #
sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)


def _save(fig, path: Path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("saved %s", path)


def plot_k_selection(scan: pd.DataFrame, k: int, out: Path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(scan.k, scan.inertia, "o-", color="#264653")
    ax[0].axvline(k, ls="--", c="#e76f51"); ax[0].set(title="Elbow (inertia)", xlabel="k", ylabel="Inertia")
    ax[1].plot(scan.k, scan.silhouette, "o-", color="#2a9d8f")
    ax[1].axvline(k, ls="--", c="#e76f51", label=f"chosen k={k}")
    ax[1].set(title="Silhouette score", xlabel="k", ylabel="Silhouette"); ax[1].legend()
    fig.tight_layout(); _save(fig, out / "01_k_selection.png")


def plot_kdist(kd: np.ndarray, eps: float, ms: int, out: Path):
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(kd, color="#264653"); ax.axhline(eps, ls="--", c="#e76f51", label=f"eps={eps:.2f}")
    ax.set(title=f"k-distance graph (k={ms})", xlabel="Points sorted by distance", ylabel=f"{ms}-NN distance")
    ax.legend(); fig.tight_layout(); _save(fig, out / "02_dbscan_kdistance.png")


def _scatter_panel(ax, emb, meta, km_labels, db_labels, label_mask, title):
    palette = sns.color_palette("tab10", int(km_labels.max()) + 1)
    noise = db_labels == -1
    for c in sorted(set(km_labels)):
        m = (km_labels == c) & ~noise
        ax.scatter(emb[m, 0], emb[m, 1], s=55, color=palette[c], edgecolor="white", lw=.6,
                   label=f"Cluster {c}")
    ax.scatter(emb[noise, 0], emb[noise, 1], s=140, marker="X", c=[palette[c] for c in km_labels[noise]],
               edgecolor="black", lw=1.1, label="DBSCAN noise", zorder=5)
    texts = [ax.text(emb[i, 0], emb[i, 1], meta.loc[i, "Label"], fontsize=7.5,
                     fontweight="bold" if noise[i] else "normal",
                     bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=.7))
             for i in np.where(label_mask)[0]]
    if adjust_text:
        adjust_text(texts, ax=ax, expand=(1.3, 1.6), arrowprops=dict(arrowstyle="-", color="grey", lw=.4))
    else:  # fallback: nudge labels off the markers
        for t in texts:
            t.set_position((t.get_position()[0] + 0.06 * np.ptp(emb[:, 0]) / 10 * 3, t.get_position()[1] + 0.12))
    ax.set_title(title)


def plot_projection(Xs, meta, km_labels, db_labels, dist_to_centroid, cfg, out: Path):
    pca = PCA(n_components=2, random_state=cfg.seed)
    p = pca.fit_transform(Xs)
    perp = max(5, min(15, (len(Xs) - 1) // 4))
    t = TSNE(n_components=2, perplexity=perp, init="pca", random_state=cfg.seed).fit_transform(Xs)

    key = ["Blade", "Acheron", "Fu Xuan", "Gepard", "Seele", "Aventurine", "Luocha", "Bronya", "Clara"]
    mask = (db_labels == -1) | meta["Character Name"].isin(key).values
    mask |= dist_to_centroid >= np.quantile(dist_to_centroid, 0.92)

    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5))
    ev = pca.explained_variance_ratio_
    _scatter_panel(axes[0], p, meta, km_labels, db_labels, mask,
                   f"PCA projection (PC1 {ev[0]:.0%}, PC2 {ev[1]:.0%})")
    _scatter_panel(axes[1], t, meta, km_labels, db_labels, mask, f"t-SNE (perplexity={perp})")
    axes[0].set(xlabel="PC1", ylabel="PC2"); axes[1].set(xlabel="t-SNE 1", ylabel="t-SNE 2")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False)
    fig.suptitle("HSR base-stat space: K-Means clusters with DBSCAN outliers (X)", fontsize=14)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97]); _save(fig, out / "03_projection_pca_tsne.png")

    loadings = pd.DataFrame(pca.components_.T, index=cfg.features, columns=["PC1", "PC2"]).round(2)
    return loadings


def plot_path_heatmap(meta, km_labels, out: Path) -> pd.DataFrame:
    ct = pd.crosstab(meta["Path"], pd.Series(km_labels, name="K-Means Cluster"))
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    sns.heatmap(ct, annot=True, fmt="d", cmap="YlOrRd", ax=ax[0], cbar=False)
    ax[0].set_title("Path × Cluster (counts)")
    sns.heatmap(ct.div(ct.sum(axis=1), axis=0), annot=True, fmt=".0%", cmap="YlGnBu", ax=ax[1], cbar=False)
    ax[1].set_title("Row-normalised: where each Path's characters land")
    fig.tight_layout(); _save(fig, out / "04_path_vs_cluster_heatmap.png")
    return ct


def plot_radar(scaled: pd.DataFrame, names: dict, out: Path):
    feats = list(scaled.columns)
    ang = np.linspace(0, 2 * np.pi, len(feats), endpoint=False).tolist(); ang += ang[:1]
    lim = max(1.0, float(np.abs(scaled.values).max()) * 1.1)
    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True))
    pal = sns.color_palette("tab10", len(scaled))
    for c, row in scaled.iterrows():
        v = (row.values + lim).tolist(); v += v[:1]          # shift so negatives plot
        ax.plot(ang, v, lw=2, color=pal[c], label=f"C{c}: {names.get(c, '')}")
        ax.fill(ang, v, alpha=.12, color=pal[c])
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(feats, fontsize=12)
    ticks = np.linspace(-lim, lim, 5)
    ax.set_yticks(ticks + lim); ax.set_yticklabels([f"{t:+.1f}σ" for t in ticks], fontsize=8)
    ax.set_ylim(0, 2 * lim)
    ax.set_title("Cluster centroids (z-scores; centre ring = dataset mean)", pad=22)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False, fontsize=9)
    _save(fig, out / "05_centroid_radar.png")


# --------------------------------------------------------------------------- #
# 5. Archetype naming (rule-based from centroid z-scores)
# --------------------------------------------------------------------------- #
def name_cluster(z: pd.Series) -> str:
    tags = []
    if "HP" in z and z["HP"] > 0.6 and z["DEF"] > 0.6: tags.append("Bulk Tank")
    elif z["DEF"] > 0.8: tags.append("Armored")
    elif z["HP"] > 0.8: tags.append("HP-Heavy")
    if z["ATK"] > 0.6 and z["DEF"] < -0.3 : tags.append("Glass Cannon")
    elif z["ATK"] > 0.6: tags.append("High-ATK")
    if z["SPD"] > 0.7: tags.append("Fast")
    elif z["SPD"] < -0.7: tags.append("Slow")
    if z["HP"] < -0.6 and z["ATK"] < -0.3: tags.append("Low-Stat Support")
    return " / ".join(tags) if tags else "Balanced Mid"


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(cfg: Config):
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    meta, X = build_dataset(cfg)
    log.info("Dataset: %d characters x %d features %s", len(X), X.shape[1], cfg.features)
    X.assign(**{"Label": meta["Label"]}).to_csv(cfg.out_dir / "features_lv80_raw.csv", index=False)

    Xs, scaler = scale_features(X)
    assert np.allclose(Xs.mean(0), 0, atol=1e-8) and np.allclose(Xs.std(0), 1, atol=1e-8)

    # ---- K-Means --------------------------------------------------------- #
    scan = scan_k(Xs, cfg); scan.to_csv(cfg.out_dir / "k_scan.csv", index=False)
    k = select_k(scan, cfg)
    km = fit_kmeans(Xs, k, cfg)
    sil = silhouette_score(Xs, km.labels_)
    scaled_c, raw_c = centroid_tables(km, scaler, cfg.features, km.labels_)
    names = {c: name_cluster(scaled_c.loc[c]) for c in scaled_c.index}
    raw_c["Archetype (auto)"] = pd.Series(names)
    log.info("K-Means: chosen k=%d (silhouette=%.3f)\n%s", k, sil, scan.round(3).to_string(index=False))

    # ---- DBSCAN ---------------------------------------------------------- #
    db, eps, kd = fit_dbscan(Xs, cfg)
    db_labels = db.labels_
    n_noise = int((db_labels == -1).sum())
    sweep = dbscan_sweep(Xs, cfg.min_samples, np.linspace(eps * 0.6, eps * 1.6, 9))
    sweep.to_csv(cfg.out_dir / "dbscan_sweep.csv", index=False)
    log.info("DBSCAN: eps=%.3f min_samples=%d -> %d clusters, %d noise\n%s", eps, cfg.min_samples,
             len(set(db_labels) - {-1}), n_noise, sweep.to_string(index=False))

    # ---- Assignment table ------------------------------------------------ #
    dist_c = np.linalg.norm(Xs - km.cluster_centers_[km.labels_], axis=1)
    trans = find_transgressors(meta, Xs, km.labels_)
    trans["DBSCAN_label"] = db_labels
    trans["Archetype"] = trans["Cluster"].map(names)
    trans["dist_to_centroid"] = dist_c.round(2)
    out_tbl = pd.concat([trans, X.round(1).reset_index(drop=True)], axis=1)
    out_tbl.to_csv(cfg.out_dir / "character_assignments.csv", index=False)
    out_tbl[out_tbl.Transgressor].sort_values("fit_margin", ascending=False) \
        .to_csv(cfg.out_dir / "role_transgressors.csv", index=False)

    audit = audit_noise(meta, X, Xs, km, db_labels, cfg.min_samples)
    audit.to_csv(cfg.out_dir / "dbscan_anomaly_audit.csv", index=False)
    scaled_c.round(3).to_csv(cfg.out_dir / "centroids_scaled.csv")
    raw_c.round(1).to_csv(cfg.out_dir / "centroids_raw.csv")

    metrics = path_alignment(meta, km.labels_)
    metrics.update({"k": k, "silhouette": sil, "eps": eps, "min_samples": cfg.min_samples,
                    "dbscan_noise": n_noise})
    pd.Series(metrics).to_csv(cfg.out_dir / "metrics.csv", header=["value"])

    # ---- Plots ----------------------------------------------------------- #
    plot_k_selection(scan, k, cfg.out_dir)
    plot_kdist(kd, eps, cfg.min_samples, cfg.out_dir)
    loadings = plot_projection(Xs, meta, km.labels_, db_labels, dist_c, cfg, cfg.out_dir)
    ct = plot_path_heatmap(meta, km.labels_, cfg.out_dir)
    plot_radar(scaled_c, names, cfg.out_dir)

    # ---- Console summary -------------------------------------------------- #
    pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_colwidth", 80)
    print("\n=== Centroids (raw Lv.80 averages) ===\n", raw_c.round(0).to_string())
    print("\n=== Centroids (z-scores) ===\n", scaled_c.round(2).to_string())
    print("\n=== PCA loadings ===\n", loadings.to_string())
    print("\n=== Path x Cluster ===\n", ct.to_string())
    print("\n=== Alignment metrics ===\n", pd.Series(metrics).round(3).to_string())
    print("\n=== DBSCAN anomaly audit ===\n", audit.to_string(index=False))
    print("\n=== Role transgressors (top by fit_margin) ===\n",
          out_tbl[out_tbl.Transgressor].sort_values("fit_margin", ascending=False)
          [["Label", "Path", "Cluster", "Archetype", "Path_modal_cluster", "Stat_implied_Path",
            "fit_margin", "DBSCAN_label"]].to_string(index=False))
    return out_tbl, audit, metrics


def parse_args() -> Config:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Config.data_dir)
    ap.add_argument("--out-dir", type=Path, default=Config.out_dir)
    ap.add_argument("--k", type=int, default=None, help="override automatic k")
    ap.add_argument("--eps", type=float, default=None, help="override DBSCAN eps (default: k-distance knee)")
    ap.add_argument("--min-samples", type=int, default=Config.min_samples)
    ap.add_argument("--include-energy", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    return Config(data_dir=a.data_dir, out_dir=a.out_dir, k_override=a.k, eps=a.eps,
                  min_samples=a.min_samples, include_energy=a.include_energy, seed=a.seed)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run(parse_args())

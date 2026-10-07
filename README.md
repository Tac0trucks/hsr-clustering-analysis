# Archetype Mining: Classifying Role Transgression in Honkai: Star Rail

An unsupervised machine learning study investigating whether character base stat allocations correspond to their assigned in-game gameplay roles (Paths) in *Honkai: Star Rail*, or if stat design is governed by latent archetypes and mechanical exceptions.

---

## Executive Summary & Core Verdict

* **Stats only weakly predict Path ($NMI = 0.25$, $ARI = 0.11$):** A character's numeric stat profile explains only about a quarter of their assigned Path structure.
* **Continuous stat space ($Silhouette = 0.28$):** Rather than forming discrete, isolated clusters for each role, characters occupy a continuous spectrum of stat distributions.
* **Design Takeaway:** Game balance in *Honkai: Star Rail* does not rely strictly on base stat differentiation. Path identity and combat roles are primarily defined through active skill kits, trace multipliers, and Light Cone synergies.

---

## Visual Deliverables

### 1. Macro Archetypes & Dimensionality Reduction (PCA & t-SNE)
K-Means ($k=4$) clusters projected into 2D space via PCA and t-SNE, highlighting DBSCAN-detected anomaly points (`X`).

![PCA & t-SNE Projections](outputs/figures/03_projection_pca_tsne.png)

### 2. Path vs. Cluster Heatmap
Cross-tabulation comparing official in-game Paths against discovered K-Means clusters. Hunt and Erudition exhibit strong alignment, while Harmony is diffuse.

![Path vs Cluster Heatmap](outputs/figures/04_path_vs_cluster_heatmap.png)

### 3. Cluster Centroid Profiles (Radar Chart)
Normalized relative stat balance ($\mu = 0, \sigma = 1$) characterizing each cluster archetype.

![Centroid Radar](outputs/figures/05_centroid_radar.png)

---

## Discovered Stat Archetypes (K-Means, $k=4$)

| Cluster | Archetype Name | Dominant Characteristics | Path Distribution |
| :---: | :--- | :--- | :--- |
| **0** | **Speed Skirmisher** | High SPD ($107$), low bulk | Captures 83% of Hunt characters. Also contains 15 of 23 four-star characters due to overall lower base stat totals. |
| **1** | **HP Sponge** | High HP ($1362$), low SPD ($97$) | HP-scaling bruisers and supports; splits Preservation tanks. |
| **2** | **Damage Core** | Highest ATK ($686$), balanced bulk | Captures 80% of Erudition units and traditional hypercarries. |
| **3** | **Armored Anchor** | Highest DEF ($656$), lowest ATK ($508$) | High-defense tanks (e.g., Aventurine, Gepard) and DEF-skewed anomalies. |

---

## Key Findings

### 1. Role Transgressions
A character is classified as a *transgressor* when their base stat profile falls outside the modal cluster of their assigned Path:
* **Firefly (Destruction):** Highest transgression margin ($1.95$). Possesses the highest base DEF ($776$) in the roster paired with low HP ($815$), placing her into the Armored Anchor cluster rather than a standard brawler profile.
* **Luocha (Abundance):** High ATK ($757$) and low DEF ($364$) place him into the Damage Core cluster alongside Erudition carries, reflecting his ATK-scaling healing mechanic.
* **Lynx (Abundance):** Clusters into Armored Anchor, exhibiting the exact inverse distribution of Luocha.
* **Harmony is Statistically Diffuse:** Harmony members are split ($4/4/2/2$) across all four clusters, indicating that buffing supports do not adhere to a single base-stat formula.

### 2. DBSCAN Anomaly Audit ($eps = 1.45, min\_samples = 4$)
Points identified as statistical noise (`label = -1`) represent extreme numerical outliers:
* **Mydei:** $2.9\times$ isolation ratio. Extreme HP ($+2.3\sigma$) paired with non-existent DEF ($-2.8\sigma$).
* **Castorice:** $1.7\times$ isolation ratio. Highest raw HP in the roster ($+2.8\sigma$).
* **Firefly:** Inverted DEF/HP coupling ($+2.7\sigma$ DEF / $-1.8\sigma$ HP).
* **Phainon:** High bulk ($+2.0\sigma$ DEF, $+1.7\sigma$ HP) coupled with very low SPD ($-1.5\sigma$).
* *Note on Blade:* Blade was **not** flagged as an anomaly; his high HP sits comfortably within the HP Sponge cluster ($0.29\sigma$ from the centroid).

---

## Repository Structure

```text
├── archetype_mining.py          # Full ingestion, clustering, and evaluation pipeline
├── requirements.txt             # Python package dependencies
├── data/
│   ├── characters.csv           # Path, element, rarity, and base SPD
│   └── character_stats.csv      # Ascension level stat curves (HP, ATK, DEF)
└── outputs/
    ├── figures/                 # Generated visualizations (PNG)
    └── tables/                  # CSV output tables (assignments, transgressors, outliers)
```

---

## Getting Started

### 1. Clone & Set Up Virtual Environment

Open your terminal and clone the repository:

```bash
git clone <YOUR-REPO-URL>
cd hsr-archetype-mining
```

Create and activate an isolated virtual environment:

- **On Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
- **On macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

### 2. Execution

Run the complete pipeline using default heuristics:

```bash
python archetype_mining.py --data-dir data/
```

### CLI Arguments & Sensitivity Checks

You can pass optional arguments to inspect alternative clustering parameters:

- `--k <int>`: Manually override optimal K-Means cluster count (e.g., `--k 5`)[cite: 3].
- `--eps <float>`: Manually set DBSCAN epsilon threshold (e.g., `--eps 1.2`)[cite: 3].
- `--min-samples <int>`: Minimum sample density for DBSCAN (default: `4`)[cite: 3].
- `--include-energy`: Run a sensitivity check incorporating imputed `Max Energy` into the standardized feature space[cite: 3].
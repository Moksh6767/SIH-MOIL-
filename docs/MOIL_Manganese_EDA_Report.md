# Exploratory Data Analysis — MOIL Manganese Ore Dataset

Source: `MOIL_Manganese_Processed_Dataset.xlsx` (built from IMYB 2024, Ch.47 Manganese Ore, and IBM Annual Report 2019-20). All charts below are generated directly from that workbook.

---

## 1. Where is production concentrated, and how is it moving?

![State production trend](eda_charts/01_state_production_trend.png)

![State share donut](eda_charts/02_state_share_donut.png)

**Reading:** Madhya Pradesh and Maharashtra together account for **61.1%** of India's 2023-24 manganese output, with Odisha a distant third (19.3%). Growth was uneven year-on-year: Andhra Pradesh grew fastest (**+39.2%**) while Telangana's output nearly halved (**-44.0%**). A national shortfall-prediction model therefore needs to treat MP and Maharashtra as the dominant — and most consequential — states: a disruption at either one moves the national number far more than an equivalent disruption anywhere else.

At the district level, **Balaghat (Madhya Pradesh)** stands out as a single point of concentration:

![District grade stacked](eda_charts/05_district_grade_stacked.png)

Balaghat alone produced **910,303 t** in 2023-24 — about **27%** of all of India's manganese ore from one district. This is also MOIL's home district, reinforcing why the problem statement centers on MOIL specifically.

---

## 2. How concentrated is production across mines? (Pareto check)

![Mine size pareto](eda_charts/06_mine_size_pareto.png)

Only **10.1%** of reporting mines (those producing 50,000+ t/year) generate **76.4%** of national output. This is a classic Pareto distribution: a shortfall-prediction model gets most of its value from monitoring a small number of large mines closely (equipment uptime, blasting schedules, weather) rather than spreading effort evenly across all 138 reporting mines.

---

## 3. Reserves vs. proved status — where should exploration effort go?

![Reserves by state](eda_charts/03_reserves_by_state.png)

Total India resources are 503.6 Mt, but only **75.0 Mt (15%)** is classified as proved *Reserves* — the rest is *Remaining Resources* (i.e., known to exist but not yet fully delineated/proved). The split is very uneven by state:

- **Madhya Pradesh** has the highest reserve "maturity" — 32.6% of its resource base is already proved.
- **Goa** holds India's 2nd-largest total resource (34.5 Mt) but only **0.2%** of it is proved reserve — almost the entire deposit is unexplored/unproved.

This is the clearest opportunity in the dataset for the AI/ML + satellite reserve-mapping component: Goa (and similarly Odisha, where 160 Mt of 171.5 Mt is still "remaining resources") represents large, low-confidence resource bases where better sub-surface/surface-indicator modeling could convert resources into de-risked, minable reserves.

---

## 4. Ore quality — how much needs beneficiation?

![Grade mix donut](eda_charts/04_grade_mix_donut.png)

Nationally, **~69%** of 2023-24 production is below 35% Mn content (low/low-medium grade). Only 7% is high-grade (46%+ Mn). This matches the narrative in IMYB 2024 that most Indian ore requires beneficiation (crushing, screening, jigging, magnetic separation) before it's usable in ferromanganese production — exactly what MOIL's Balaghat and Dongri Buzurg beneficiation plants are for. Any production-planning model should treat "raw tonnes extracted" and "usable/high-grade tonnes" as two different target variables, since the gap between them is large and grade-dependent.

---

## 5. Inventory build-up — is production actually reaching the market?

![Closing stock by state](eda_charts/07_closing_stock_by_state.png)

Mine-head closing stock rose in **every** reporting state in 2023-24 — most sharply in Rajasthan (+368%) and Andhra Pradesh (+114%). Stock is building up faster than production is growing in most states, which is a useful (and slightly counter-intuitive) signal for the shortfall-prediction module: a "shortfall" isn't only under-production — unsold/unmoved stock piling up at the mine-head signals a logistics- or demand-side bottleneck rather than a mining-side one, and the two need different corrective actions (the problem statement's "re-deploying equipment" fix won't help a stock-pileup problem; better rail/road logistics scheduling would).

---

## 6. India in world context — why does India import so much?

![World reserves](eda_charts/08_world_reserves.png)

![World production trend](eda_charts/09_world_production_trend.png)

![Trade balance](eda_charts/10_trade_balance.png)

India holds only **~1.8%** of world manganese reserves (rank #8, 34,000 kt of 1,900,000 kt metal content) but contributes roughly **6%** of world production — i.e., India is extracting a disproportionately large share of its resource base relative to peers like South Africa or Australia, who sit on much larger reserve cushions. India's own production grew at a healthy **~10.5% CAGR** (2021-2023), yet the country still imports about **20,600x more** manganese ore by volume than it exports (5.59 Mt imports vs. 271 t exports in 2023-24). This import-dependency, despite being a top-6 global producer, is the direct real-world motivation behind the problem statement: better reserve discovery and shortfall avoidance domestically could reduce this import gap.

---

## 7. What do the ground-truth exploration samples actually show?

![Exploration Mn ranges](eda_charts/11_exploration_mn_ranges.png)

The 2023-24 exploration round-up (used to build the `Exploration_Indicators` sheet) shows wide within-site variability in assayed Mn content — e.g., the Seoni & Balaghat district survey found grades ranging from near-zero up to 35.8% Mn within the same reconnaissance area. This heterogeneity is exactly the challenge a spatial ML model needs to handle: sparse borehole/sample points with high local variance, which is why the problem statement calls for *surface and sub-surface indicators* (soil moisture, vegetation index, land temperature, etc.) as additional covariates to interpolate between sample points rather than relying on drilling density alone.

---

## Summary: what this means for the ML solution design

| Observation | Implication for the AI/ML solution |
|---|---|
| 61% of production from 2 states, 27% from 1 district (Balaghat) | Prioritize high-resolution monitoring/modeling in MP & Maharashtra, especially Balaghat, before scaling nationally |
| 10% of mines produce 76% of output | Equipment-downtime and blasting-delay features matter most for the ~14 large mines, not all 138 |
| Goa & Odisha have large unproved resources | Best target areas for the satellite-based reserve-identification module — highest potential payoff per unit of new data |
| ~69% of output is low/medium grade | Model raw tonnes and beneficiation-adjusted (usable) tonnes separately |
| Stock is rising in every state | Include inventory/logistics signals, not just production volume, when defining "shortfall" |
| India: 1.8% of world reserves, ~6% of world production, imports >20,000x its exports | Frames the business case — even modest gains in domestic reserve discovery or shortfall avoidance meaningfully reduce import dependency |
| Exploration assays show high local variance | Justifies using satellite/geospatial covariates (rainfall, soil moisture, NDVI, LST) to interpolate between sparse ground samples, as the problem statement specifies |

**Caveat carried over from the dataset build:** this EDA is based entirely on the IBM/IMYB statistical and exploration-survey publications. It does not include actual satellite time-series (rainfall, soil moisture, vegetation index, land temperature) or live equipment/blasting logs — those still need to be sourced (ISRO Bhuvan, Sentinel-2/MODIS, Landsat/MODIS LST, MOIL's internal SCADA/ERP) and joined to this data by state/district/mine location before building the predictive model itself.

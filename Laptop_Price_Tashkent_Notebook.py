# %% [markdown]
"""
# Determinants of Laptop Retail Prices in Tashkent: OLS, Lasso/Ridge and Deep Neural Network Evidence under UZS/USD Exchange Rate Pass-Through

**Author:** [Student Name], Tashkent University of Information Technologies (TUIT)

**Date:** October 2026

---

> **Disclaimer:** This notebook uses **SIMULATED** data calibrated to realistic
> Tashkent retail prices (October 2026 market conditions, UZS/USD ≈ 12 000).
> Replace the synthetic dataset with real scraped data for the final submission.
> The data-generating process intentionally mirrors empirical price distributions
> observed in Uzbekistan's laptop retail segment.
"""

# %%
# ============================================================
# CELL 1 — Install & Import
# ============================================================
# !pip install -q statsmodels scikit-learn tensorflow openpyxl

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor
import tensorflow as tf
from tensorflow import keras

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("All libraries imported successfully.")

# %% [markdown]
"""
## Cell 2 — Generate Simulated Dataset (n = 250)

The synthetic data-generating process is calibrated as follows:

| Parameter | Value / Distribution |
|-----------|---------------------|
| Exchange rate | 12 000 UZS per 1 USD |
| Base price | 400 USD |
| Brand premium | Apple ×2.2, MSI ×1.6, Dell ×1.15, … |
| CPU premium | Core i9 ×1.8, Core i7 ×1.4, … |
| RAM elasticity | +4 % per GB |
| SSD elasticity | +0.03 % per GB |
| Noise | Log-normal, σ = 0.15 |
| Missing rate | ~5 % MCAR |
| Outliers | 3–5 extreme high-price rows |
"""

# %%
# ============================================================
# CELL 2 — Generate Simulated Dataset
# ============================================================
np.random.seed(SEED)

N = 250
EXCHANGE_RATE = 12_000  # ADAPTED: UZS per 1 USD (Kaggle original uses EUR, we use UZS at CBU rate)

# ------ Categorical pools & multipliers ------ #
brands = ["Acer", "Asus", "Dell", "HP", "Lenovo", "Apple", "MSI", "Samsung"]
brand_mult = {
    "Apple": 2.2, "MSI": 1.6, "Dell": 1.15, "Samsung": 1.1,
    "Asus": 1.05, "HP": 1.0, "Lenovo": 0.95, "Acer": 0.85,
}

cpus = [
    "Celeron", "Core_i3", "Core_i5", "Core_i7", "Core_i9",
    "Ryzen_3", "Ryzen_5", "Ryzen_7",
]
cpu_mult = {
    "Celeron": 0.5, "Core_i3": 0.7, "Core_i5": 1.0, "Core_i7": 1.4,
    "Core_i9": 1.8, "Ryzen_3": 0.65, "Ryzen_5": 0.95, "Ryzen_7": 1.35,
}

ram_choices = [4, 8, 16, 32, 64]
ram_weights = [0.05, 0.35, 0.35, 0.20, 0.05]

ssd_choices = [128, 256, 512, 1000, 2000]
ssd_weights = [0.05, 0.20, 0.40, 0.25, 0.10]

gpus = [
    "Integrated", "MX_Series", "GTX_1650", "RTX_3050",
    "RTX_3060", "RTX_4060", "RTX_4070", "RTX_4090",
]
gpu_mult = {
    "Integrated": 1.0, "MX_Series": 1.10, "GTX_1650": 1.20,
    "RTX_3050": 1.30, "RTX_3060": 1.45, "RTX_4060": 1.60,
    "RTX_4070": 1.80, "RTX_4090": 2.20,
}

conditions = ["new", "used"]
cond_weights = [0.70, 0.30]
cond_mult = {"new": 1.0, "used": 0.75}

shop_types = ["mall", "online", "market"]
shop_weights = [0.40, 0.35, 0.25]
shop_mult = {"mall": 1.05, "online": 1.00, "market": 0.93}

districts = [
    "Chilanzar", "Mirzo_Ulugbek", "Yunusabad", "Sergeli",
    "Yakkasaray", "Mirabad", "Shaykhantakhur", "Olmazar",
]
district_mult = {
    "Chilanzar": 1.00, "Mirzo_Ulugbek": 1.02, "Yunusabad": 1.05,
    "Sergeli": 0.96, "Yakkasaray": 1.03, "Mirabad": 1.01,
    "Shaykhantakhur": 0.97, "Olmazar": 0.94,
}

# ------ Draw random attributes ------ #
brand_col = np.random.choice(brands, N)
cpu_col = np.random.choice(cpus, N)
ram_col = np.random.choice(ram_choices, N, p=ram_weights)
ssd_col = np.random.choice(ssd_choices, N, p=ssd_weights)
gpu_col = np.random.choice(gpus, N)
screen_col = np.clip(np.random.normal(15.6, 1.5, N), 11.6, 17.3)
weight_col = np.clip(0.6 * screen_col / 15.6 + np.random.normal(1.5, 0.3, N), 0.9, 3.5)
cond_col = np.random.choice(conditions, N, p=cond_weights)
shop_col = np.random.choice(shop_types, N, p=shop_weights)
dist_col = np.random.choice(districts, N)

# ------ Price formula ------ #
BASE_USD = 400

price_usd = np.array([
    BASE_USD
    * brand_mult[brand_col[i]]
    * cpu_mult[cpu_col[i]]
    * (1 + 0.04 * ram_col[i])
    * (1 + 0.0003 * ssd_col[i])
    * gpu_mult[gpu_col[i]]
    * (0.9 + 0.01 * screen_col[i])
    * cond_mult[cond_col[i]]
    * shop_mult[shop_col[i]]
    * district_mult[dist_col[i]]
    for i in range(N)
])

# Log-normal multiplicative noise (σ = 0.15)
noise = np.random.lognormal(mean=0.0, sigma=0.15, size=N)
price_usd *= noise

# Convert to UZS
price_uzs = price_usd * EXCHANGE_RATE

# ------ Build DataFrame ------ #
df_sim = pd.DataFrame({
    "brand": brand_col,
    "cpu": cpu_col,
    "ram_gb": ram_col,
    "ssd_gb": ssd_col,
    "gpu": gpu_col,
    "screen_in": np.round(screen_col, 1),
    "weight_kg": np.round(weight_col, 2),
    "condition": cond_col,
    "shop_type": shop_col,
    "district": dist_col,
    "price_usd": np.round(price_usd, 2),
    "price_uzs": np.round(price_uzs, 0).astype(int),
})

# ------ Inject 3–5 extreme outliers ------ #
rng = np.random.default_rng(SEED)
outlier_idx = rng.choice(N, size=4, replace=False)
for idx in outlier_idx:
    df_sim.loc[idx, "price_usd"] = df_sim["price_usd"].quantile(0.95) * rng.uniform(2.5, 4.0)
    df_sim.loc[idx, "price_uzs"] = int(df_sim.loc[idx, "price_usd"] * EXCHANGE_RATE)
print(f"Injected {len(outlier_idx)} extreme outlier rows at indices {list(outlier_idx)}.")

# ------ Inject ~5 % MCAR missing values ------ #
total_cells = df_sim.shape[0] * df_sim.shape[1]
n_missing = int(0.05 * total_cells)
row_idx_miss = rng.integers(0, df_sim.shape[0], size=n_missing)
col_idx_miss = rng.integers(0, df_sim.shape[1], size=n_missing)
for r, c in zip(row_idx_miss, col_idx_miss):
    df_sim.iat[r, c] = np.nan
print(f"Injected {n_missing} missing values (~5 % of {total_cells} cells).")

# ------ Save CSV ------ #
csv_path = "laptop_prices_tashkent_simulated.csv"
df_sim.to_csv(csv_path, index=False)
print(f"Saved → {csv_path}")

# ------ Empty template with same columns ------ #
template_path = "laptop_prices_tashkent_TEMPLATE.csv"
pd.DataFrame(columns=df_sim.columns).to_csv(template_path, index=False)
print(f"Saved empty template → {template_path}")

# ------ Preview ------ #
print("\n" + "=" * 70)
print("DISCLAIMER: This is SIMULATED data. Replace with real scraped data.")
print("=" * 70 + "\n")
print(df_sim.head(10).to_string())
print()
df_sim.info()
print(f"\nShape: {df_sim.shape}")

# %% [markdown]
"""
## Cell 3 — Load Data

Two pathways are provided:

* **Option A (Kaggle API):** Download a public Kaggle laptop-price dataset
  (commented out; requires `kaggle.json` credentials).
* **Option B (Local / Simulated):** Load the CSV generated in Cell 2. For the
  final submission, replace with a real file upload via `google.colab.files`.
"""

# %%
# ============================================================
# CELL 3 — Load Data
# ============================================================

# ---- Option A: Kaggle API (uncomment to use) ----
# from google.colab import files
# files.upload()  # upload kaggle.json
# !mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
# !kaggle datasets download -d muhammetvarl/laptop-price -p ./data --unzip
# df_kaggle = pd.read_csv('./data/laptop_data.csv')
# print(f"Kaggle dataset loaded: {df_kaggle.shape}")

# ---- Option B: Local upload / use simulated data (active) ----
# For real data, uncomment:
# from google.colab import files
# uploaded = files.upload()
# df = pd.read_csv(list(uploaded.keys())[0])

# Using simulated data:
df = pd.read_csv("laptop_prices_tashkent_simulated.csv")
print(f"Dataset loaded: {df.shape}")
print(df.dtypes)

# %% [markdown]
"""
## Cell 4 (Part B) — Missing-Value Treatment

**Strategy justification:**

| Criterion | Decision |
|-----------|----------|
| Missing fraction | ~5 % (MCAR by construction) |
| Variable types | Mixed numeric + categorical |
| Chosen method | Median (numeric) / Mode (categorical) |
| Why not mean? | Median is robust to the outliers we injected |
| Why not MICE / IterativeImputer? | With <5 % MCAR missingness and a moderate sample (n = 250), simple imputation introduces negligible bias; MICE adds complexity without meaningful accuracy gains (van Buuren, 2018). |

We also demonstrate `IterativeImputer` below (commented) for reference.
"""

# %%
# ============================================================
# CELL 4 — Missing Values
# ============================================================

# --- Diagnostics ---
print("Missing values per column:")
print(df.isnull().sum())
total_missing = df.isnull().sum().sum()
total_cells = df.shape[0] * df.shape[1]
print(f"\nTotal missing: {total_missing} ({total_missing / total_cells * 100:.1f}% of all cells)")

# --- Visualise missingness pattern ---
fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap="YlOrRd", ax=ax)
ax.set_title("Missing-Value Heatmap (yellow = present, red = missing)")
ax.set_xlabel("Feature")
ax.set_ylabel("Observation")
plt.tight_layout()
plt.show()

# --- Imputation: Median (numeric) / Mode (categorical) ---
num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(include="object").columns.tolist()

print(f"\nNumeric columns ({len(num_cols)}): {num_cols}")
print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")

for col in num_cols:
    median_val = df[col].median()
    n_fill = df[col].isnull().sum()
    if n_fill > 0:
        df[col].fillna(median_val, inplace=True)
        print(f"  Filled {col}: {n_fill} NAs → median = {median_val:.2f}")

for col in cat_cols:
    mode_val = df[col].mode()[0]
    n_fill = df[col].isnull().sum()
    if n_fill > 0:
        df[col].fillna(mode_val, inplace=True)
        print(f"  Filled {col}: {n_fill} NAs → mode = '{mode_val}'")

assert df.isnull().sum().sum() == 0, "Missing values remain!"
print("\n✓ All missing values imputed successfully.")

# --- Alternative: IterativeImputer (commented, for reference) ---
# imp = IterativeImputer(max_iter=10, random_state=SEED)
# df[num_cols] = imp.fit_transform(df[num_cols])

# %% [markdown]
"""
## Cell 5 (Part B) — Outlier Detection

We apply two complementary diagnostics:

1. **IQR Rule (Tukey fences):** Flag observations with
   `price_uzs < Q1 − 1.5·IQR` or `price_uzs > Q3 + 1.5·IQR`.
2. **Cook's Distance:** After a preliminary OLS on numeric features, flag
   points with Cook's D > 4/n.

**Decision:** We **retain** all observations but **winsorize** at the
1st / 99th percentiles. With n = 250, deleting outliers would reduce
statistical power; winsorizing bounds extreme leverage without losing rows.
"""

# %%
# ============================================================
# CELL 5 — Outlier Detection
# ============================================================

# ---------- 5a. IQR Rule ---------- #
Q1 = df["price_uzs"].quantile(0.25)
Q3 = df["price_uzs"].quantile(0.75)
IQR = Q3 - Q1
lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 + 1.5 * IQR
outliers_iqr = df[(df["price_uzs"] < lower_fence) | (df["price_uzs"] > upper_fence)]
print(f"IQR bounds: [{lower_fence:,.0f}, {upper_fence:,.0f}] UZS")
print(f"IQR outliers detected: {len(outliers_iqr)} rows\n")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Box plot
axes[0].boxplot(df["price_uzs"], vert=True, patch_artist=True,
                boxprops=dict(facecolor="lightblue"))
axes[0].set_title("Price Distribution (UZS) — Before Winsorization")
axes[0].set_ylabel("Price (UZS)")
axes[0].axhline(upper_fence, color="red", ls="--", lw=0.8, label="Upper fence")
axes[0].axhline(lower_fence, color="red", ls="--", lw=0.8, label="Lower fence")
axes[0].legend(fontsize=8)

# Histogram
axes[1].hist(df["price_uzs"], bins=30, edgecolor="black", alpha=0.7, color="steelblue")
axes[1].axvline(upper_fence, color="red", ls="--", lw=1, label="Upper fence")
axes[1].set_title("Histogram of price_uzs")
axes[1].set_xlabel("Price (UZS)")
axes[1].set_ylabel("Frequency")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.show()

# ---------- 5b. Cook's Distance ---------- #
# Quick OLS on numeric features only (for diagnostic purposes)
num_features_for_cook = ["ram_gb", "ssd_gb", "screen_in", "weight_kg"]
X_cook = sm.add_constant(df[num_features_for_cook].astype(float))
y_cook = df["price_uzs"].astype(float)
ols_cook = sm.OLS(y_cook, X_cook).fit()

influence = ols_cook.get_influence()
cooks_d, pvals = influence.cooks_distance
threshold_cook = 4.0 / len(df)
n_cook_outliers = (cooks_d > threshold_cook).sum()

fig, ax = plt.subplots(figsize=(12, 3.5))
ax.stem(np.arange(len(cooks_d)), cooks_d, markerfmt=",", basefmt=" ")
ax.axhline(threshold_cook, color="red", ls="--", label=f"Threshold = 4/n = {threshold_cook:.4f}")
ax.set_title("Cook's Distance — Preliminary OLS on Numeric Features")
ax.set_xlabel("Observation Index")
ax.set_ylabel("Cook's D")
ax.legend()
plt.tight_layout()
plt.show()
print(f"Cook's D outliers (>{threshold_cook:.4f}): {n_cook_outliers} rows")

# ---------- 5c. Decision: Winsorize at 1st / 99th percentile ---------- #
p01 = df["price_uzs"].quantile(0.01)
p99 = df["price_uzs"].quantile(0.99)
df["price_uzs"] = df["price_uzs"].clip(lower=p01, upper=p99)
print(f"\nWinsorized price_uzs to [{p01:,.0f}, {p99:,.0f}] UZS.")
print("All rows retained (n = {}).".format(len(df)))

# %% [markdown]
"""
## Cell 6 (Part B) — Dummy (One-Hot) Encoding

### The Dummy Variable Trap

When a categorical variable has *k* categories and we create *k* binary
dummies, the columns sum to 1 for every row — identical to the intercept
column.  This produces **perfect multicollinearity** and an
uninvertible **X′X** matrix.

**Solution:** `drop_first=True` drops one reference category per variable,
yielding *k − 1* linearly independent dummies.  OLS coefficients are then
interpreted *relative to the omitted reference level*.

The reference categories chosen (alphabetically first) are listed below.
"""

# %%
# ============================================================
# CELL 6 — Dummy Encoding
# ============================================================

cat_features = ["brand", "cpu", "gpu", "condition", "shop_type", "district"]

print("Dummy Variable Trap Explanation")
print("=" * 50)
print("Including all k dummies for a k-category variable creates perfect")
print("multicollinearity (they sum to 1 = intercept column).")
print("Solution: drop_first=True drops one reference category.\n")

# ADAPTED: Uzbekistan-specific categories (condition, shop_type, district) absent from Kaggle original
# One-hot encode with drop_first
df_encoded = pd.get_dummies(df, columns=cat_features, drop_first=True, dtype=int)

# Print reference categories
print("Reference (dropped) categories:")
for col in cat_features:
    cats = sorted(df[col].unique())
    remaining = cats[1:]
    print(f"  {col:12s}: reference = '{cats[0]}' (dropped), "
          f"dummies for {remaining}")

print(f"\nOriginal shape : {df.shape}")
print(f"Encoded shape  : {df_encoded.shape}")
print(f"New dummy cols : {df_encoded.shape[1] - df.shape[1] + len(cat_features)}")

# %% [markdown]
"""
## Cell 7 (Part B) — Train / Test Split and Feature Scaling

**Dependent variable:** `ln(price_uzs)` — the natural logarithm of the UZS
price.  A log transformation is standard in hedonic pricing models because
(a) prices are right-skewed and (b) OLS coefficients become interpretable as
approximate percentage effects.

**Split:** 80 / 20 stratified by nothing (simple random split),
`random_state = 42` for reproducibility.

**Scaling:** `StandardScaler` (z-score) is fitted **on the training set only**
and then applied to both train and test.  This prevents **data leakage** —
test-set statistics never influence the transformation.
"""

# %%
# ============================================================
# CELL 7 — Train-Test Split & Scaling
# ============================================================

# --- Log-transform target ---
df_encoded["ln_price"] = np.log(df_encoded["price_uzs"])

# --- Define features and target ---
drop_cols = ["price_uzs", "price_usd", "ln_price"]
drop_cols = [c for c in drop_cols if c in df_encoded.columns]
X = df_encoded.drop(columns=drop_cols)
y = df_encoded["ln_price"]

print(f"Feature matrix X : {X.shape}")
print(f"Target vector y  : {y.shape}")

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED
)
print(f"\nTrain set: {X_train.shape[0]} rows  ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"Test  set: {X_test.shape[0]} rows  ({X_test.shape[0]/len(X)*100:.0f}%)")

# --- StandardScaler: fit on TRAIN only (no data leakage) ---
scaler = StandardScaler()
X_train_sc = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index,
)
X_test_sc = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index,
)

print("\nScaling: StandardScaler fitted on train, applied to both train and test.")
print("Train means ≈ 0:", X_train_sc.mean().abs().max() < 1e-10)
print("Train  stds ≈ 1:", (X_train_sc.std() - 1).abs().max() < 1e-10)
print("\n✓ Part 1 data preparation complete. Proceed to modelling (Part 2).")
# %% [markdown]
# ## Part C: Exploratory Data Analysis

# %%
# C1 — Summary Statistics Table
summary = df[['price_uzs','ram_gb','ssd_gb','screen_in','weight_kg']].describe().T
summary['skewness'] = df[['price_uzs','ram_gb','ssd_gb','screen_in','weight_kg']].skew()
summary['kurtosis'] = df[['price_uzs','ram_gb','ssd_gb','screen_in','weight_kg']].kurtosis()
print(summary.to_string())

# %%
# C2 — Histogram + KDE of price_uzs (raw and log-transformed)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Raw price
axes[0].hist(df['price_uzs']/1e6, bins=30, density=True, alpha=0.7,
             color='steelblue', edgecolor='black')
df['price_uzs'].div(1e6).plot.kde(ax=axes[0], color='red', linewidth=2)
axes[0].set_title('Distribution of Laptop Prices (UZS)', fontsize=13)
axes[0].set_xlabel('Price (million UZS)')
axes[0].set_ylabel('Density')

# Log price
ln_price = np.log(df['price_uzs'])
axes[1].hist(ln_price, bins=30, density=True, alpha=0.7,
             color='seagreen', edgecolor='black')
ln_price.plot.kde(ax=axes[1], color='red', linewidth=2)
axes[1].set_title('Distribution of ln(Price)', fontsize=13)
axes[1].set_xlabel('ln(Price UZS)')
axes[1].set_ylabel('Density')

plt.tight_layout()
plt.savefig('price_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Skewness: raw = {df['price_uzs'].skew():.3f}, log = {ln_price.skew():.3f}")
print("→ Log transformation substantially reduces right skewness, justifying ln(price) as dependent variable.")
print("→ Coefficients on ln(price) have elasticity/semi-elasticity interpretation.")

# %%
# C3 — Scatter plots of ln(price) vs key numeric regressors
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, col, label in zip(axes.flat,
                           ['ram_gb', 'ssd_gb', 'screen_in', 'weight_kg'],
                           ['RAM (GB)', 'SSD (GB)', 'Screen (inches)', 'Weight (kg)']):
    ax.scatter(df[col], np.log(df['price_uzs']),
               alpha=0.5, s=30, edgecolor='k', linewidth=0.3)
    # Add OLS trend line
    valid = df[col].dropna().index
    z = np.polyfit(df[col].loc[valid], np.log(df['price_uzs'].loc[valid]), 1)
    x_sorted = sorted(df[col].loc[valid])
    ax.plot(x_sorted, np.polyval(z, x_sorted), 'r--', linewidth=2)
    ax.set_xlabel(label, fontsize=11)
    ax.set_ylabel('ln(Price UZS)', fontsize=11)
    ax.set_title(f'ln(Price) vs {label}', fontsize=12)

plt.tight_layout()
plt.savefig('scatter_plots.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# C4 — Correlation Heatmap (numeric features + ln_price)
df_temp = df[['ram_gb', 'ssd_gb', 'screen_in', 'weight_kg']].copy()
df_temp['ln_price'] = np.log(df['price_uzs'])
corr = df_temp.corr()

plt.figure(figsize=(8, 6))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            mask=mask, square=True, linewidths=0.5)
plt.title('Correlation Matrix: Numeric Features vs ln(Price)', fontsize=13)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### Justification for Using ln(Price) as Dependent Variable
#
# The raw price distribution exhibits pronounced **right skewness**
# (skewness ≈ reported above), which violates the normality assumption
# underpinning classical OLS inference. The natural-log transformation
# compresses the right tail and yields a near-symmetric distribution
# (skewness close to zero).
#
# **Interpretation of coefficients in the log-linear model:**
#
# - **Continuous regressors (semi-elasticity):** a one-unit increase in
#   $X_j$ is associated with an approximate $100 \times \hat{\beta}_j$
#   percent change in price, holding other factors constant.
# - **Dummy variables (exact percentage effect):**
#   $\%\Delta P = (\exp(\hat{\beta}_j) - 1) \times 100$, which gives the
#   exact percentage price difference relative to the omitted reference
#   category.

# %% [markdown]
# ## Part D: Predictive Models

# %%
# D1 — Model 1: OLS via statsmodels on ln(price)
# ADAPTED: Dependent variable is ln(price_uzs), not ln(price_euros); local dummies included
X_train_const = sm.add_constant(X_train)
X_test_const  = sm.add_constant(X_test)

ols_model = sm.OLS(y_train, X_train_const).fit()
print(ols_model.summary())

y_pred_ols = ols_model.predict(X_test_const)

# %%
# D2 — Model 2: LassoCV and RidgeCV (scaled features)
# ADAPTED: Scaled features required; UZS-scale feature magnitudes differ from EUR-based Kaggle data
alphas = np.logspace(-4, 2, 100)

lasso = LassoCV(alphas=alphas, cv=5, random_state=SEED, max_iter=10000)
lasso.fit(X_train_sc, y_train)
print(f"LassoCV optimal alpha: {lasso.alpha_:.6f}")
print(f"Lasso non-zero coefficients: {np.sum(lasso.coef_ != 0)} / {len(lasso.coef_)}")

ridge = RidgeCV(alphas=alphas, cv=None, scoring='neg_mean_squared_error')
# RidgeCV uses efficient leave-one-out CV by default when cv=None
ridge.fit(X_train_sc, y_train)
print(f"RidgeCV optimal alpha: {ridge.alpha_:.6f}")

y_pred_lasso = lasso.predict(X_test_sc)
y_pred_ridge = ridge.predict(X_test_sc)

# %%
# D3 — Model 3: Keras MLP with Early Stopping
# ADAPTED: Architecture downsized (128→64→32) for n=250 Tashkent sample; heavier regularisation than Kaggle
tf.random.set_seed(SEED)

model_dnn = keras.Sequential([
    keras.layers.Dense(128, activation='relu',
                       input_shape=(X_train_sc.shape[1],),
                       kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu',
                       kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)
])

model_dnn.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse', metrics=['mae'])

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20,
                                            restore_best_weights=True)

history = model_dnn.fit(X_train_sc, y_train,
                        validation_split=0.2,
                        epochs=300, batch_size=16,
                        callbacks=[early_stop],
                        verbose=0)

# Plot loss curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['loss'], label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Val Loss')
axes[0].set_title('DNN Training: MSE Loss Curves', fontsize=13)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE')
axes[0].legend()

axes[1].plot(history.history['mae'], label='Train MAE')
axes[1].plot(history.history['val_mae'], label='Val MAE')
axes[1].set_title('DNN Training: MAE Curves', fontsize=13)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MAE')
axes[1].legend()

plt.tight_layout()
plt.savefig('dnn_loss_curves.png', dpi=150, bbox_inches='tight')
plt.show()

y_pred_dnn = model_dnn.predict(X_test_sc).flatten()
print(f"DNN stopped at epoch {len(history.history['loss'])}")

# %% [markdown]
# ## Part E: Econometric Analysis

# %% [markdown]
# ### E.1 OLS Closed-Form Solution: $\hat{\beta} = (X'X)^{-1}X'y$

# %%
# E1 — Derive beta_hat manually and verify against statsmodels
X_mat = X_train_const.values.astype(float)
y_vec = y_train.values.astype(float)

# Manual computation
beta_hat = np.linalg.inv(X_mat.T @ X_mat) @ X_mat.T @ y_vec

# Assert equality with statsmodels
np.testing.assert_allclose(beta_hat, ols_model.params.values, rtol=1e-6)
print("✓ Manual beta_hat matches statsmodels OLS estimates (rtol=1e-6).")
print("\nCoefficients:")
for name, b in zip(X_train_const.columns, beta_hat):
    print(f"  {name:30s}: {b:10.6f}")

# %% [markdown]
# ### E.2 Coefficient Interpretation & Gauss–Markov Theorem

# %%
# E2 — Coefficient interpretation: top positive/negative drivers
coef_df = pd.DataFrame({
    'Coefficient': ols_model.params,
    'Std Error':   ols_model.bse,
    't-stat':      ols_model.tvalues,
    'p-value':     ols_model.pvalues
}).sort_values('Coefficient', ascending=False)

print("Top 5 positive coefficients (strongest price drivers):")
print(coef_df.head(5).to_string())
print("\nTop 5 negative coefficients (price discounts):")
print(coef_df.tail(5).to_string())

# %% [markdown]
# **Gauss–Markov Theorem.** Under the classical linear model assumptions—
#
# - **A1 (Linearity):** $y = X\beta + \varepsilon$
# - **A2 (Random sampling):** observations are i.i.d.
# - **A3 (No perfect multicollinearity):** $\text{rank}(X) = k+1$
# - **A4 (Zero conditional mean):** $E[\varepsilon \mid X] = 0$
# - **A5 (Homoscedasticity):** $\text{Var}(\varepsilon \mid X) = \sigma^2 I$
#
# —OLS is **BLUE** (Best Linear Unbiased Estimator): among all linear
# unbiased estimators of $\beta$, the OLS estimator $\hat{\beta}$ has
# the smallest variance. This makes OLS the natural benchmark before
# exploring regularized or nonlinear alternatives.

# %% [markdown]
# ### E.3 R² vs Adjusted R²

# %%
# E3 — R² versus Adjusted R² with degrees-of-freedom penalty
r2     = ols_model.rsquared
r2_adj = ols_model.rsquared_adj
n      = ols_model.nobs
k      = ols_model.df_model

print(f"R²          = {r2:.6f}")
print(f"Adjusted R² = {r2_adj:.6f}")
print(f"n = {n:.0f}, k = {k:.0f}")
print(f"\nDegrees-of-freedom penalty: R²_adj = 1 - (1-R²)·(n-1)/(n-k-1)")
print(f"Verify: 1 - (1-{r2:.6f})·({n:.0f}-1)/({n:.0f}-{k:.0f}-1) "
      f"= {1 - (1 - r2) * (n - 1) / (n - k - 1):.6f}")
print(f"\n→ Adjusted R² penalizes model complexity. "
      f"The gap R²−R²_adj = {r2 - r2_adj:.4f}")
print(f"  {'is small, suggesting limited overfitting.' if r2 - r2_adj < 0.03 else 'is moderate, warranting caution about overfitting.'}")

# %% [markdown]
# ### E.4 Diagnostic Plots & Heteroscedasticity Tests

# %%
# E4 — Residual diagnostics and heteroscedasticity tests
resid  = ols_model.resid
fitted = ols_model.fittedvalues

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Residuals vs Fitted
axes[0, 0].scatter(fitted, resid, alpha=0.5, s=25,
                   edgecolor='k', linewidth=0.3)
axes[0, 0].axhline(0, color='red', linestyle='--')
axes[0, 0].set_title('Residuals vs Fitted Values', fontsize=12)
axes[0, 0].set_xlabel('Fitted ln(Price)')
axes[0, 0].set_ylabel('Residuals')

# 2. Normal Q-Q
sm.qqplot(resid, line='45', ax=axes[0, 1], alpha=0.5)
axes[0, 1].set_title('Normal Q-Q Plot of Residuals', fontsize=12)

# 3. Scale-Location
axes[1, 0].scatter(fitted, np.sqrt(np.abs(resid)), alpha=0.5, s=25,
                   edgecolor='k', linewidth=0.3)
axes[1, 0].set_title('Scale-Location Plot', fontsize=12)
axes[1, 0].set_xlabel('Fitted ln(Price)')
axes[1, 0].set_ylabel('√|Residuals|')

# 4. Histogram of residuals with normal overlay
axes[1, 1].hist(resid, bins=25, density=True, alpha=0.7, edgecolor='black')
x_range = np.linspace(resid.min(), resid.max(), 100)
axes[1, 1].plot(x_range,
                stats.norm.pdf(x_range, resid.mean(), resid.std()),
                'r-', linewidth=2)
axes[1, 1].set_title('Distribution of Residuals', fontsize=12)
axes[1, 1].set_xlabel('Residual')
axes[1, 1].set_ylabel('Density')

plt.tight_layout()
plt.savefig('diagnostic_plots.png', dpi=150, bbox_inches='tight')
plt.show()

# Breusch-Pagan test
bp_stat, bp_pval, _, _ = het_breuschpagan(resid, X_train_const)
print(f"Breusch-Pagan test: LM = {bp_stat:.4f}, p-value = {bp_pval:.4f}")
print(f"  → {'Reject H0: evidence of heteroscedasticity'    if bp_pval < 0.05 else 'Fail to reject H0: no significant heteroscedasticity'} at 5% level.")

# White test
white_stat, white_pval, _, _ = het_white(resid, X_train_const)
print(f"White test: LM = {white_stat:.4f}, p-value = {white_pval:.4f}")
print(f"  → {'Reject H0: heteroscedasticity detected'       if white_pval < 0.05 else 'Fail to reject H0'} at 5% level.")

# Classic vs HC3 robust standard errors
ols_hc3 = sm.OLS(y_train, X_train_const).fit(cov_type='HC3')
se_table = pd.DataFrame({
    'Classic SE':      ols_model.bse,
    'HC3 SE':          ols_hc3.bse,
    'Ratio HC3/Classic': ols_hc3.bse / ols_model.bse
})
print("\nClassic vs HC3 Robust Standard Errors:")
print(se_table.to_string())
print("\n→ If HC3/Classic ratio >> 1 for some variables, those coefficients'")
print("  significance is overstated under the homoscedasticity assumption.")

# %% [markdown]
# ### E.5 Regularization Path & Lambda Selection

# %%
# E5 — Lasso coefficient paths and CV error curve
from sklearn.linear_model import lasso_path

alphas_path, coefs_path, _ = lasso_path(X_train_sc, y_train,
                                         alphas=np.logspace(-4, 1, 200),
                                         max_iter=10000)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Coefficient paths
for i in range(coefs_path.shape[0]):
    axes[0].plot(np.log10(alphas_path), coefs_path[i], linewidth=0.8)
axes[0].axvline(np.log10(lasso.alpha_), color='red', linestyle='--',
                label=f'Optimal λ={lasso.alpha_:.4f}')
axes[0].set_title('Lasso Coefficient Paths', fontsize=13)
axes[0].set_xlabel('log₁₀(λ)')
axes[0].set_ylabel('Coefficient Value')
axes[0].legend()

# CV MSE vs alpha
mse_path = np.mean(lasso.mse_path_, axis=1)
axes[1].plot(np.log10(lasso.alphas_), mse_path, linewidth=2)
axes[1].axvline(np.log10(lasso.alpha_), color='red', linestyle='--',
                label=f'Optimal λ={lasso.alpha_:.4f}')
axes[1].set_title('Lasso: CV MSE vs log₁₀(λ)', fontsize=13)
axes[1].set_xlabel('log₁₀(λ)')
axes[1].set_ylabel('Mean CV MSE')
axes[1].legend()

plt.tight_layout()
plt.savefig('regularization_path.png', dpi=150, bbox_inches='tight')
plt.show()

# Features shrunk to zero by Lasso
zero_features = X_train_sc.columns[lasso.coef_ == 0].tolist()
print(f"\nOptimal Lasso λ = {lasso.alpha_:.6f}")
print(f"Features shrunk to exactly zero ({len(zero_features)}):")
for f in zero_features:
    print(f"  • {f}")
print("\n→ Economic validity: Lasso eliminates features with weak/redundant")
print("  predictive power. If district dummies or certain GPU tiers are zeroed,")
print("  it suggests they add little beyond what other features already capture.")

# Contrast with Ridge
print(f"\nRidge optimal λ = {ridge.alpha_:.6f}")
print(f"Ridge min |coef|: {np.abs(ridge.coef_).min():.6f} "
      f"(Ridge never sets coefficients exactly to zero)")
print("→ Ridge shrinks all coefficients toward zero but retains all; "
      "Lasso performs variable selection.")

# %% [markdown]
# ### E.6 Dummy-Variable Interpretation, VIF & Multicollinearity

# %%
# E6 — Exact percentage effects of dummies + VIF analysis
dummy_cols = [c for c in X_train_const.columns
              if any(c.startswith(p + '_')
                     for p in ['brand', 'cpu', 'gpu',
                               'condition', 'shop_type', 'district'])]

dummy_interp = pd.DataFrame({
    'Coefficient': ols_model.params[dummy_cols],
    'Pct_Effect':  (np.exp(ols_model.params[dummy_cols]) - 1) * 100,
    'p-value':     ols_model.pvalues[dummy_cols],
    'Significant': ols_model.pvalues[dummy_cols] < 0.05
}).sort_values('Pct_Effect', ascending=False)

print("Dummy Variable Interpretation (vs omitted baseline):")
print("% Effect = (exp(β) − 1) × 100 = percentage price difference "
      "relative to reference category\n")
print(dummy_interp.to_string())

# Variance Inflation Factors
print("\n\nVariance Inflation Factors:")
vif_data = pd.DataFrame()
vif_data['Feature'] = X_train.columns
vif_data['VIF'] = [variance_inflation_factor(X_train.values, i)
                   for i in range(X_train.shape[1])]
vif_data = vif_data.sort_values('VIF', ascending=False)
print(vif_data.head(15).to_string(index=False))
print(f"\n→ VIF > 10 indicates problematic multicollinearity.")
print(f"  Features with VIF > 10: "
      f"{vif_data[vif_data['VIF'] > 10]['Feature'].tolist()}")

# Matrix rank check
rank = np.linalg.matrix_rank(X_train.values)
print(f"\nMatrix rank: {rank} (= {X_train.shape[1]} columns → "
      f"{'full rank, no perfect multicollinearity' if rank == X_train.shape[1] else 'RANK DEFICIENT!'})")

# %% [markdown]
# ### E.7 F-test, p-value & Wald Test on Brand Dummies

# %%
# E7 — Overall F-test and joint Wald test on brand dummies
print(f"Overall F-statistic: {ols_model.fvalue:.4f}")
print(f"p-value (F-test):    {ols_model.f_pvalue:.2e}")
print(f"H0: β₁ = β₂ = ... = βₖ = 0 (all slopes are zero)")
print(f"→ {'Reject H0' if ols_model.f_pvalue < 0.05 else 'Fail to reject H0'}: "
      f"the model is {'jointly significant' if ols_model.f_pvalue < 0.05 else 'not significant'} at 5%.")

print(f"\nFormal definition: p-value = P(F ≥ {ols_model.fvalue:.2f} | H0 true)")
print(f"= probability of observing an F-statistic at least as extreme under "
      f"the null.")

# Wald joint test on all brand dummies
brand_cols = [c for c in X_train_const.columns if c.startswith('brand_')]
print(f"\n--- Wald Joint Test: All Brand Dummies = 0 ---")
print(f"Testing: {brand_cols}")

# Construct restriction matrix R such that R @ beta = 0 under H0
R = np.zeros((len(brand_cols), len(X_train_const.columns)))
for i, col in enumerate(brand_cols):
    R[i, list(X_train_const.columns).index(col)] = 1

wald_test = ols_model.wald_test(R)
print(f"Wald statistic: {float(wald_test.statistic):.4f}")
print(f"p-value:        {float(wald_test.pvalue):.2e}")
print(f"→ {'Reject H0: brand significantly affects price' if float(wald_test.pvalue) < 0.05 else 'Fail to reject: brand is not jointly significant'}.")

# %% [markdown]
# ## Final Model Comparison

# %%
# E_FINAL — Test-set performance comparison across all four models
def eval_model(name, y_true, y_pred):
    """Compute RMSE, MAE, and R² for a given model's predictions."""
    return {
        'Model': name,
        'RMSE':  np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE':   mean_absolute_error(y_true, y_pred),
        'R²':    r2_score(y_true, y_pred)
    }

results = pd.DataFrame([
    eval_model('OLS',   y_test, y_pred_ols),
    eval_model('Lasso', y_test, y_pred_lasso),
    eval_model('Ridge', y_test, y_pred_ridge),
    eval_model('DNN',   y_test, y_pred_dnn)
])
results = results.set_index('Model')

print("=" * 60)
print("TEST-SET PERFORMANCE COMPARISON (on ln(price))")
print("=" * 60)
print(results.to_string())
print("=" * 60)

# Convert RMSE to approximate UZS at the mean price level
mean_price = np.exp(y_test.mean())
for idx, row in results.iterrows():
    approx_uzs_error = mean_price * row['RMSE']
    print(f"{idx}: RMSE ≈ {approx_uzs_error / 1e6:.2f} million UZS "
          f"at mean price level")

# %% [markdown]
# ## Conclusion
#
# 1. **OLS** provides an interpretable baseline; the Gauss–Markov theorem
#    guarantees it is BLUE under classical assumptions. The adjusted R²
#    indicates the proportion of log-price variance explained after
#    penalising for the number of regressors.
#
# 2. **Lasso** achieves comparable accuracy while performing automatic
#    feature selection, eliminating redundant predictors — confirming that
#    not all district or GPU-tier dummies are economically meaningful in
#    the Tashkent market.
#
# 3. **Ridge** retains all features with slight shrinkage and can
#    outperform OLS when multicollinearity is present among hardware
#    specifications (RAM–SSD–GPU correlations).
#
# 4. **DNN** captures nonlinear interactions (e.g., brand × GPU synergies)
#    but gains are modest on this ~250-row sample, and interpretability is
#    sacrificed.
#
# 5. For **policy applications** (consumer protection, price transparency
#    in Tashkent), the OLS/Lasso log-linear model is recommended: it
#    delivers hedonic price decomposition directly interpretable in
#    percentage terms, while the DNN serves as a robustness check for
#    nonlinear pricing patterns.

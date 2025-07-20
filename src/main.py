import yfinance as yf
import cot_reports as cot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from scipy import stats

# 1. Fetch WTI Price Data (Yahoo Finance)
print("Fetching WTI Crude Oil prices from Yahoo Finance...")
ticker = yf.Ticker("CL=F")
wti_price = ticker.history(start="2015-01-01", end="2025-02-24")
wti_price.index = wti_price.index.tz_localize(None)
wti_price_weekly = wti_price.resample("W-SUN").mean()
wti_price_weekly["Close"].to_csv("data/wti_prices.csv")
print("Price data saved to data/wti_prices.csv")

# 2. Fetch COT Data
years = range(2015, 2025)
cot_dfs = []
print("Fetching COT data...")
for year in years:
    print(f"  -> {year}")
    try:
        yearly_data = cot.cot_year(cot_report_type='legacy_fut', year=year)
        cot_dfs.append(yearly_data)
    except Exception as e:
        print(f"Failed to fetch {year}: {e}")

if not cot_dfs:
    print("No COT data fetched — exiting.")
    exit(1)

cot_data = pd.concat(cot_dfs, ignore_index=True)
wti_cot = cot_data[cot_data["CFTC Contract Market Code"] == "067651"].copy()
wti_cot["As of Date in Form YYYY-MM-DD"] = pd.to_datetime(wti_cot["As of Date in Form YYYY-MM-DD"], utc=False)
wti_cot.set_index("As of Date in Form YYYY-MM-DD", inplace=True)

# Enrich COT with features
wti_cot["Comm_Share_Long"] = wti_cot["Commercial Positions-Long (All)"] / wti_cot["Open Interest (All)"]
wti_cot["Comm_Share_Short"] = wti_cot["Commercial Positions-Short (All)"] / wti_cot["Open Interest (All)"]
wti_cot["NonComm_Share_Long"] = wti_cot["Noncommercial Positions-Long (All)"] / wti_cot["Open Interest (All)"]
wti_cot["NonComm_Share_Short"] = wti_cot["Noncommercial Positions-Short (All)"] / wti_cot["Open Interest (All)"]
wti_cot["Net_Spec_Position"] = wti_cot["Noncommercial Positions-Long (All)"] - wti_cot["Noncommercial Positions-Short (All)"]
wti_cot["Net_Comm_Position"] = wti_cot["Commercial Positions-Long (All)"] - wti_cot["Commercial Positions-Short (All)"]

for col in ["Open Interest (All)", "Noncommercial Positions-Long (All)", "Commercial Positions-Short (All)",
            "Noncommercial Positions-Short (All)", "Commercial Positions-Long (All)"]:
    wti_cot[f"{col}_Change"] = wti_cot[col].pct_change()

wti_cot.to_csv("data/wti_cot_enriched.csv")
print("Enriched COT data saved to data/wti_cot_enriched.csv")

# Resample COT to weekly (Sunday end)
wti_cot_weekly = wti_cot.resample("W-SUN").last()

# 3. Merge for Price Direction
print("Merging COT and price for direction...")
merged_df = pd.merge(wti_cot_weekly, wti_price_weekly[["Close"]], left_index=True, right_index=True, how="inner")
if merged_df.empty:
    print("Merged DataFrame is empty. Check date alignment.")
    print("COT Date Range:", wti_cot_weekly.index.min(), "to", wti_cot_weekly.index.max())
    print("Price Date Range:", wti_price_weekly.index.min(), "to", wti_price_weekly.index.max())
    exit(1)

# Define price direction (up/down) next week
merged_df["Price_Change"] = merged_df["Close"].shift(-1) - merged_df["Close"]
merged_df["Direction"] = np.where(merged_df["Price_Change"] > 0, 1, -1)  # 1 = up, -1 = down
merged_df.dropna(inplace=True)

# Drop price data after defining direction
wti_cot_final = merged_df.drop(columns=["Close", "Price_Change"])
numeric_cols = wti_cot_final.select_dtypes(include=[np.number]).columns
wti_cot_final = wti_cot_final[numeric_cols]
wti_cot_final.to_csv("data/wti_cot_final.csv")
print("Final COT data with direction saved to data/wti_cot_final.csv")

# 4. Deep ML Analysis for Correlations and Patterns
print("Performing deep ML analysis for correlations and patterns...")

# Prepare features and target
features = [col for col in wti_cot_final.columns if col != "Direction"]
X = wti_cot_final[features]
y = wti_cot_final["Direction"]

# Train-test split (time-ordered, no shuffle)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 4.1. Random Forest for Feature Importance (Correlations)
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

# Feature importance (correlations)
rf_importance = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print("\nRandom Forest Feature Importances (Correlations):")
print(rf_importance)
plt.figure(figsize=(12, 6))
rf_importance.plot(kind='barh', color='teal')
plt.title("Random Forest Feature Importances for WTI COT-Price Correlations")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("data/rf_feature_importance.png")
plt.show()

# 4.2. XGBoost for Deeper Correlations and Interactions
xgb = XGBClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
print("\nXGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))
print("\nXGBoost Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))

# XGBoost feature importance (correlations)
xgb_importance = pd.Series(xgb.feature_importances_, index=features).sort_values(ascending=False)
print("\nXGBoost Feature Importances (Correlations):")
print(xgb_importance)
plt.figure(figsize=(12, 6))
xgb_importance.plot(kind='barh', color='purple')
plt.title("XGBoost Feature Importances for WTI COT-Price Correlations")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("data/xgb_feature_importance.png")
plt.show()

# 4.3. K-means Clustering for Pattern Discovery
# Normalize features for clustering
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Try K-means with different clusters (2–5) to find patterns
for k in range(2, 6):
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    wti_cot_final[f'Cluster_{k}'] = clusters
    print(f"\nK-means Clustering (k={k}) Patterns:")
    cluster_summary = wti_cot_final.groupby(f'Cluster_{k}')[features].mean()
    print(cluster_summary)
    plt.figure(figsize=(12, 6))
    cluster_summary.plot(kind='bar', stacked=True, colormap='viridis')
    plt.title(f"WTI COT Patterns by Cluster (k={k})")
    plt.xlabel("Cluster")
    plt.ylabel("Average Feature Value")
    plt.tight_layout()
    plt.savefig(f"data/cluster_patterns_k{k}.png")
    plt.show()

# 4.4. Statistical Correlations (Pearson and Spearman)
print("\nStatistical Correlations (Pearson):")
pearson_corr = wti_cot_final[features + ["Direction"]].corr()
print(pearson_corr["Direction"].sort_values(ascending=False))

print("\nStatistical Correlations (Spearman):")
spearman_corr = wti_cot_final[features + ["Direction"]].corr(method='spearman')
print(spearman_corr["Direction"].sort_values(ascending=False))

# 4.5. Interaction Analysis (Pairwise Correlations)
print("\nTop 10 Pairwise Feature Interactions (Pearson):")
pairwise_corr = wti_cot_final[features].corr()
pairwise_corr = pairwise_corr.unstack().sort_values(absolute=True, ascending=False)
pairwise_corr = pairwise_corr[pairwise_corr.index.get_level_values(0) != pairwise_corr.index.get_level_values(1)].drop_duplicates()
print(pairwise_corr.head(10))

# 4.6. Output All Patterns and Rules
print("\nDetailed ML Findings and Potential Trading Rules:")
print("- Feature Importances (RF/XGBoost): Use top features like 'Noncommercial Positions-Short (All)_Change' and 'Noncommercial Positions-Long (All)_Change' for signals.")
print("- Clusters: Look for clusters where 'Net_Spec_Position' or 'Comm_Share_Long' peaks—buy in bullish clusters, sell in bearish.")
print("- Statistical Correlations: Buy when 'Noncommercial Positions-Long (All)_Change' > 0.05 (positive Spearman), sell when 'Commercial Positions-Short (All)_Change' < -0.05 (negative Spearman).")
print("- Pairwise Interactions: Exploit strong interactions like 'Noncommercial Positions-Long (All)_Change' vs. 'Open Interest (All)_Change' for combined signals.")

# Save all correlations and patterns
rf_importance.to_csv("data/rf_correlations.csv")
xgb_importance.to_csv("data/xgb_correlations.csv")
pearson_corr.to_csv("data/pearson_correlations.csv")
spearman_corr.to_csv("data/spearman_correlations.csv")
pairwise_corr.head(10).to_csv("data/pairwise_correlations.csv")
for k in range(2, 6):
    wti_cot_final.to_csv(f"data/clusters_k{k}.csv")

print("All correlations and patterns saved to data/ directory.")

# 5. Momentum for Additional Insights
print("Calculating momentum (WaveTrend-style)...")
n1, n2 = 10, 21
ap = wti_price_weekly[["High", "Low", "Close"]].mean(axis=1)
esa = ap.ewm(span=n1, adjust=False).mean()
d = (ap - esa).abs().ewm(span=n1, adjust=False).mean()
ci = (ap - esa) / (0.015 * d)
tci = ci.ewm(span=n2, adjust=False).mean()
wt1 = tci
wt2 = wt1.rolling(4).mean()
wti_price_weekly["WT1"] = wt1
wti_price_weekly["WT2"] = wt2
momentum_buy = (wt1 > wt2) and (wt1 < -60)
momentum_sell = (wt1 < wt2) and (wt1 > 60)
wti_price_weekly["Momentum_Buy"] = momentum_buy
wti_price_weekly["Momentum_Sell"] = momentum_sell

wti_price_weekly[["WT1", "WT2", "Momentum_Buy", "Momentum_Sell"]].to_csv("data/wti_momentum.csv")
print("Momentum signals saved to data/wti_momentum.csv")

# 6. Final Trading Signals (for Pine Script)
print("Generating trading signals based on ML findings...")
trading_signals = pd.DataFrame(index=wti_price_weekly.index)
trading_signals["ML_Score"] = rf.predict_proba(wti_cot_final[features])[:, 1]
trading_signals["Momentum_Buy"] = wti_price_weekly["Momentum_Buy"]
trading_signals["Momentum_Sell"] = wti_price_weekly["Momentum_Sell"]

# Use top ML correlations for signals (example rules from findings)
top_features = rf_importance.head(5).index.tolist()
trading_signals["Top_Feature_Score"] = X[top_features].mean(axis=1) * rf_importance.head(5).values

buy_signal = (trading_signals["ML_Score"] > 0.70) and trading_signals["Momentum_Buy"] and (trading_signals["Top_Feature_Score"] > trading_signals["Top_Feature_Score"].quantile(0.8))
sell_signal = ((1 - trading_signals["ML_Score"]) > 0.4053) and trading_signals["Momentum_Sell"] and (trading_signals["Top_Feature_Score"] < trading_signals["Top_Feature_Score"].quantile(0.2))

trading_signals["Buy_Signal"] = buy_signal
trading_signals["Sell_Signal"] = sell_signal

trading_signals[["Buy_Signal", "Sell_Signal"]].to_csv("data/trading_signals.csv")
print("Trading signals saved to data/trading_signals.csv")
print("Deep ML analysis complete. Check data/ for detailed outputs.")
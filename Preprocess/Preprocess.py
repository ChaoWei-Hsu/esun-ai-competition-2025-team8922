"""
Processes financial transaction data for fraud detection modeling.

This script executes Part 1 of a machine learning pipeline: Feature Engineering.
It performs the following steps:
1.  Loads account transaction and alert data from CSV files.
2.  Engineers a comprehensive set of features for each account, including:
    - Temporal features (e.g., late-night activity, lifespan)
    - Aggregate in-degree/out-degree statistics (e.g., sum, mean, count)
    - Advanced temporal features (e.g., trend, volatility)
    - Anomaly scores (e.g., transaction Z-scores)
    - Graph-native features (e.g., PageRank)
    - Expert-driven interaction features (e.g., hub score, risk-weighted volume)
3.  Scales all engineered features using StandardScaler.
4.  Saves the final feature tensor, account-to-index mapping, feature names,
    and the fitted scaler to a PyTorch file (`.pt`).

This script is designed to be run as the first step before training a
Graph Neural Network (GNN) or other model.
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from sklearn.preprocessing import StandardScaler
import time
import gc
import networkx as nx

### --- 0. GPU Detection ---
print("--- Part 1: Feature Engineering (Expert Optimized & Corrected + Interactions) ---")
print("Step 0: Checking for GPU...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"GPU '{torch.cuda.get_device_name(0)}' is available.")
else:
    print("No GPU found. Running on CPU.")

### --- 1. Data Loading ---
print("\nStep 1: Loading data...")
try:
    # ⚠️ Ensure these paths are correct for your local environment.
    df_trans = pd.read_csv("data/acct_transaction.csv", dtype={'from_acct': str, 'to_acct': str})
    df_alert = pd.read_csv("data/acct_alert.csv", dtype={'acct': str})
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    # ⚠️ Please check that the file paths above are correct!
    print("⚠️ Please check that the file paths above are correct!")
    exit()

### --- 2. Feature Engineering ---
print("\nStep 2: Performing expert feature engineering...")
tqdm.pandas()

# Basic temporal features
df_trans['txn_hour'] = pd.to_datetime(df_trans['txn_time'], format='%H:%M:%S', errors='coerce').dt.hour
df_trans['is_late_night'] = ((df_trans['txn_hour'] >= 0) & (df_trans['txn_hour'] < 6)).astype(int)
df_trans['txn_date'] = pd.to_datetime(df_trans['txn_date'], errors='coerce')

# Categorical features
df_trans = pd.get_dummies(df_trans, columns=['channel_type'], prefix='channel', dtype=int)

# Amount features
df_trans['is_round_number_txn'] = (df_trans['txn_amt'] % 1000 == 0).astype(int)

print("Calculating out-degree features...")
out_agg_funcs = {
    'txn_amt': ['sum', 'mean', 'std', 'max', 'count'],
    'to_acct': ['nunique'], 'txn_date': ['min', 'max', 'nunique'],
    'is_late_night': ['sum', 'mean'], 'is_round_number_txn': ['mean'],
    # Dynamically add 'sum' aggregation for all one-hot encoded channel_ columns
    **{col: 'sum' for col in df_trans.columns if 'channel_' in col}
}
out_features = df_trans.groupby('from_acct').agg(out_agg_funcs)
out_features.columns = ['_'.join(col).strip() for col in out_features.columns.values]
out_features = out_features.rename(columns={
    'txn_amt_sum': 'total_out_amount', 'txn_amt_count': 'total_out_txns', 
    'txn_amt_mean': 'avg_out_amount', 'txn_amt_std': 'std_out_amount', 
    'txn_amt_max': 'max_out_amount', 'to_acct_nunique': 'unique_to_accts', 
    'txn_date_min': 'first_out_txn_date', 'txn_date_max': 'last_out_txn_date', 
    'txn_date_nunique': 'unique_out_txn_days', 'is_late_night_sum': 'late_night_out_txn_count', 
    'is_late_night_mean': 'late_night_out_txn_ratio', 
    'is_round_number_txn_mean': 'round_number_out_txn_ratio'
})

# Convert timestamp features into a numerical 'lifespan' feature.
print("Converting timestamp features to numerical 'account_lifespan'...")
# Calculate the lifespan in days
out_features['account_lifespan_days'] = (out_features['last_out_txn_date'] - out_features['first_out_txn_date']).dt.days
# Drop the original non-numerical timestamp columns
out_features = out_features.drop(columns=['first_out_txn_date', 'last_out_txn_date'])


print("Calculating in-degree features...")
in_agg_funcs = {'txn_amt': ['sum', 'mean', 'count'], 'from_acct': ['nunique']}
in_features = df_trans.groupby('to_acct').agg(in_agg_funcs)
in_features.columns = ['in_' + '_'.join(col).strip() for col in in_features.columns.values]
in_features = in_features.rename(columns={
    'in_txn_amt_sum': 'total_in_amount', 'in_txn_amt_count': 'total_in_txns', 
    'in_txn_amt_mean': 'avg_in_amount', 'in_from_acct_nunique': 'unique_from_accts'
})

print("Extracting advanced temporal features...")
def extract_advanced_features(sub_df):
    """
    Calculates advanced temporal features for a single account's transactions.

    This function is intended to be applied to a pandas groupby object
    (e.g., `df.groupby('from_acct').progress_apply(extract_advanced_features)`).

    It calculates:
    - 'activity_density': Ratio of active days to total account lifespan.
    - 'amt_trend': The slope of a linear regression fit to daily total transaction amounts.
    - 'growth_rate': The percentage change from the first day's amount to the last.
    - 'amt_volatility': The standardized standard deviation of daily amount changes.
    - 'count_volatility': The standardized standard deviation of daily transaction count changes.

    Args:
        sub_df (pd.DataFrame): A DataFrame subset containing all transactions
                               for a single 'from_acct'. It must contain
                               'txn_date' and 'txn_amt' columns.

    Returns:
        pd.Series: A Series indexed by the feature names
                   ('amt_trend', 'amt_volatility', etc.) holding the
                   calculated values for the account. Returns zeros for
                   accounts with insufficient data (e.g., < 2 active days).
    """
    sub_df = sub_df.sort_values('txn_date')
    daily_stats = sub_df.groupby('txn_date').agg(daily_amt_sum=('txn_amt', 'sum'), daily_txn_count=('txn_amt', 'count'))
    
    activity_span_days = (sub_df['txn_date'].max() - sub_df['txn_date'].min()).days + 1
    activity_density = daily_stats.shape[0] / activity_span_days if activity_span_days > 0 else 0
    
    if len(daily_stats) < 2:
        return pd.Series({'amt_trend': 0, 'amt_volatility': 0, 'growth_rate': 0, 'count_volatility': 0, 'activity_density': activity_density})
        
    y_amt = daily_stats['daily_amt_sum'].values; x = np.arange(len(y_amt))
    try: 
        amt_trend = np.polyfit(x, y_amt, 1)[0]
    except (np.linalg.LinAlgError, ValueError): 
        amt_trend = 0
        
    growth_rate = (y_amt[-1] - y_amt[0]) / (y_amt[0] + 1e-9)
    amt_volatility = np.std(np.diff(y_amt)) / (np.mean(y_amt) + 1e-9)
    
    y_count = daily_stats['daily_txn_count'].values
    count_volatility = np.std(np.diff(y_count)) / (np.mean(y_count) + 1e-9) if len(y_count) > 1 else 0
    
    return pd.Series({
        'amt_trend': amt_trend, 'amt_volatility': amt_volatility, 
        'growth_rate': growth_rate, 'count_volatility': count_volatility, 
        'activity_density': activity_density
    })
adv_temporal_features = df_trans.groupby('from_acct').progress_apply(extract_advanced_features)

print("Calculating behavioral anomaly Z-scores...")
stats = df_trans.groupby('from_acct')['txn_amt'].agg(['mean', 'std']).rename(columns={'mean': 'avg_out_amount_for_z', 'std': 'std_out_amount_for_z'})
df_trans = df_trans.join(stats, on='from_acct')
epsilon = 1e-9 # Avoid division by zero
df_trans['txn_zscore'] = (df_trans['txn_amt'] - df_trans['avg_out_amount_for_z']) / (df_trans['std_out_amount_for_z'].fillna(0) + epsilon)
max_zscore_features = df_trans.groupby('from_acct').agg(max_txn_zscore=('txn_zscore', 'max'))

print("Calculating graph-native features (PageRank)...")
G = nx.from_pandas_edgelist(df_trans, 'from_acct', 'to_acct', create_using=nx.DiGraph())
pagerank = nx.pagerank(G, alpha=0.85)
pagerank_features = pd.DataFrame(list(pagerank.items()), columns=['acct_id', 'pagerank']).set_index('acct_id')

print("Integrating all features...")
all_accts = pd.unique(np.concatenate([df_trans['from_acct'], df_trans['to_acct']]))
all_accts_df = pd.DataFrame(index=all_accts)
node_features = all_accts_df.join(out_features).join(in_features).join(adv_temporal_features).join(max_zscore_features).join(pagerank_features)
node_features.fillna(0, inplace=True)

print("Calculating ratio and liquidity features...")
node_features['net_flow'] = node_features['total_in_amount'] - node_features['total_out_amount']
node_features['in_out_amount_ratio'] = node_features['total_in_amount'] / (node_features['total_out_amount'] + epsilon)
node_features['max_avg_out_ratio'] = node_features['max_out_amount'] / (node_features['avg_out_amount'] + epsilon)

# =============================================================================
# Interaction Features
# =============================================================================
print("Adding interaction features...")

# 1. Hub Score: unique_from_accts * unique_to_accts (Identifies intermediary accounts)
node_features['hub_score'] = node_features['unique_from_accts'] * node_features['unique_to_accts']

# 2. Risk-Weighted Volume: late_night_out_txn_ratio * total_out_amount
node_features['risk_weighted_volume'] = node_features['late_night_out_txn_ratio'] * node_features['total_out_amount']

# 3. Volatility-Weighted Volume: amt_volatility * total_out_amount
node_features['volatility_weighted_volume'] = node_features['amt_volatility'] * node_features['total_out_amount']

# 4. PageRank-Weighted Flow: pagerank * net_flow (Identifies important sources or sinks)
node_features['pagerank_weighted_flow'] = node_features['pagerank'] * node_features['net_flow']

# 5. Normalized Daily Activity (per day of account lifespan)
node_features['avg_daily_out_txns'] = node_features['total_out_txns'] / (node_features['account_lifespan_days'] + epsilon)
node_features['avg_daily_out_amount'] = node_features['total_out_amount'] / (node_features['account_lifespan_days'] + epsilon)
node_features['avg_daily_in_txns'] = node_features['total_in_txns'] / (node_features['account_lifespan_days'] + epsilon)

# 6. Anomaly Magnitude: max_txn_zscore * avg_out_amount
node_features['anomaly_magnitude'] = node_features['max_txn_zscore'] * node_features['avg_out_amount']

# =============================================================================

# Final cleanup
node_features.replace([np.inf, -np.inf], 0, inplace=True)
node_features.fillna(0, inplace=True)
# Reporting the total number of features created
print(f"✅ Final feature count: {node_features.shape[1]}") 

del out_features, in_features, adv_temporal_features, max_zscore_features, pagerank_features, all_accts_df, df_trans, G, stats
gc.collect()

### --- 3. Feature Scaling ---
print("\nStep 3: Scaling node features...")
scaler = StandardScaler()
scaled_features_np = scaler.fit_transform(node_features)
node_features_scaled = pd.DataFrame(scaled_features_np, index=node_features.index, columns=node_features.columns)

### --- 4. Saving Processed Features ---
print("\nStep 4: Saving processed features to file...")
acct_to_idx = {acct_id: i for i, acct_id in enumerate(node_features_scaled.index)}
feature_tensor = torch.tensor(node_features_scaled.values, dtype=torch.float)
# Set the output filename
save_path = 'processed_features_with_interactions.pt' 
torch.save({
    'feature_tensor': feature_tensor, 
    'acct_to_idx': acct_to_idx, 
    'feature_names': list(node_features.columns), # Save feature names for later analysis
    'scaler': scaler
}, save_path)
print(f"✅ Features saved to '{save_path}'")
print("\n--- Part 1 Completed ---")
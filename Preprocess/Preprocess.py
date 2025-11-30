import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from sklearn.preprocessing import StandardScaler
import time
import gc
from scipy import sparse

"""
Financial Transaction Feature Engineering Pipeline

This script performs high-performance, vectorized feature engineering on financial 
transaction data. It generates:
1. Basic aggregation features (sum, mean, std, counts).
2. Advanced temporal features (linear trends, volatility, density).
3. Network/Graph features (PageRank, degrees) using sparse matrices.
4. Interaction features (ratios, net flows).
5. Registered/Whitelist features (Static connection counts and transactional verification).

The output is a normalized PyTorch tensor suitable for Graph Neural Networks 
or Tabular models.
"""

# Configuration
pd.options.mode.chained_assignment = None

def fast_pagerank(A, d=0.85, tol=1e-4, max_iter=20):
    """
    Calculates PageRank using a sparse adjacency matrix and power iteration.
    
    Args:
        A (scipy.sparse.coo_matrix): Adjacency matrix of the graph.
        d (float): Damping factor.
        tol (float): Convergence tolerance.
        max_iter (int): Maximum number of iterations.
        
    Returns:
        np.array: PageRank scores for all nodes.
    """
    n = A.shape[0]
    
    # Normalize matrix: Row stochastic
    out_degree = np.array(A.sum(axis=1)).flatten()
    out_degree[out_degree == 0] = 1.0 
    D_inv = sparse.diags(1.0 / out_degree)
    M = D_inv @ A 
    
    # Initialization
    r = np.ones(n) / n
    teleport = np.ones(n) / n
    
    for _ in range(max_iter):
        r_new = d * (M.T @ r) + (1 - d) * teleport
        if np.linalg.norm(r_new - r, 1) < tol:
            break
        r = r_new
    return r

### --- 0. GPU Detection ---
print("--- Part 1: Feature Engineering ---")
print("Step 0: Checking for GPU...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"GPU '{torch.cuda.get_device_name(0)}' is available.")
else:
    print("No GPU found. Running on CPU.")

# --- 1. Data Loading ---
print("\nStep 1: Loading data...")
try:
    # Attempt to use pyarrow engine for faster IO, fallback to standard C engine on failure
    try:
        df_trans = pd.read_csv("data/acct_transaction.csv", dtype={'from_acct': str, 'to_acct': str}, engine='pyarrow')
        df_alert = pd.read_csv("data/acct_alert.csv", dtype={'acct': str}, engine='pyarrow')
        # [NEW] Load Register Data
        df_reg = pd.read_csv("data/acct_register.csv", dtype={'from_acct': str, 'to_acct': str}, engine='pyarrow')
    except:
        print("Pyarrow engine not found, using default...")
        df_trans = pd.read_csv("data/acct_transaction.csv", dtype={'from_acct': str, 'to_acct': str})
        df_alert = pd.read_csv("data/acct_alert.csv", dtype={'acct': str})
        # [NEW] Load Register Data
        df_reg = pd.read_csv("data/acct_register.csv", dtype={'from_acct': str, 'to_acct': str})
        
    print("Data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    exit()

# --- 2. Feature Engineering ---
print("\nStep 2: Performing vectorized feature engineering...")

# Pre-processing temporal columns
df_trans['txn_time'] = pd.to_datetime(df_trans['txn_time'], format='%H:%M:%S', errors='coerce')
df_trans['txn_hour'] = df_trans['txn_time'].dt.hour
df_trans['is_late_night'] = ((df_trans['txn_hour'] >= 0) & (df_trans['txn_hour'] < 6)).astype(int)
df_trans['txn_date'] = pd.to_datetime(df_trans['txn_date'], errors='coerce')

# Convert dates to numeric index 
# Note: Aligning with register data where Day 1 is the start.
min_date = df_trans['txn_date'].min()
# We add +1 so the first day is 1, matching the register file logic
df_trans['day_idx'] = (df_trans['txn_date'] - min_date).dt.days + 1 

# One-hot encoding and logic checks
df_trans = pd.get_dummies(df_trans, columns=['channel_type'], prefix='channel', dtype=int)
df_trans['is_round_number_txn'] = (df_trans['txn_amt'] % 1000 == 0).astype(int)

# ==========================================
# [NEW] Processing Register (Whitelist) Logic
# ==========================================
print("Processing Register (Whitelist) Features...")

# 1. Clean Register Data (Clamp dates as per instructions)
max_txn_day = df_trans['day_idx'].max()
df_reg['start_date'] = pd.to_numeric(df_reg['start_date'], errors='coerce').fillna(-1).astype(int)
df_reg['end_date'] = pd.to_numeric(df_reg['end_date'], errors='coerce').fillna(999).astype(int)

# Logic: start_date < 1 -> -1, end_date > max -> 999
df_reg.loc[df_reg['start_date'] < 1, 'start_date'] = -1
df_reg.loc[df_reg['end_date'] > max_txn_day, 'end_date'] = 999

# 2. Static Register Features (Trust Graph Structure)
# Count how many beneficiaries this account has registered (Trusted Out-degree)
reg_static_out = df_reg.groupby('from_acct')['to_acct'].nunique().to_frame('num_registered_beneficiaries')
# Count how many people have registered this account (Trusted In-degree)
reg_static_in = df_reg.groupby('to_acct')['from_acct'].nunique().to_frame('num_registered_sources')

# 3. Dynamic Transaction Flagging
# Merge transactions with register data to verify if specific txns are whitelisted
# We perform a left join on (from, to) and then check the date window
print("   -> Merging register data with transactions...")
df_trans_reg = df_trans[['from_acct', 'to_acct', 'day_idx']].merge(
    df_reg[['from_acct', 'to_acct', 'start_date', 'end_date']],
    on=['from_acct', 'to_acct'],
    how='left'
)

# A transaction is registered if the link exists AND the date is within the window
df_trans_reg['is_registered_txn'] = (
    (df_trans_reg['start_date'].notna()) & 
    (df_trans_reg['day_idx'] >= df_trans_reg['start_date']) & 
    (df_trans_reg['day_idx'] <= df_trans_reg['end_date'])
).astype(int)

# Assign flag back to main dataframe and clean up
df_trans['is_registered_txn'] = df_trans_reg['is_registered_txn']
del df_trans_reg, df_reg # Free memory

# --- Vectorized Out-degree Features ---
print("Calculating out-degree features (Vectorized)...")

out_agg_funcs = {
    'txn_amt': ['sum', 'mean', 'std', 'max', 'count'],
    'to_acct': ['nunique'], 
    'day_idx': ['min', 'max', 'nunique'], 
    'is_late_night': ['sum', 'mean'], 
    'is_round_number_txn': ['mean'],
    'is_registered_txn': ['sum', 'mean'] # [NEW] Added registered stats
}

# Dynamically add channel columns to aggregation
for col in df_trans.columns:
    if 'channel_' in col:
        out_agg_funcs[col] = ['sum']

out_features = df_trans.groupby('from_acct').agg(out_agg_funcs)
out_features.columns = ['_'.join(col).strip() for col in out_features.columns.values]

# Calculate account lifespan and rename columns for clarity
out_features['account_lifespan_days'] = out_features['day_idx_max'] - out_features['day_idx_min']
out_features = out_features.rename(columns={
    'txn_amt_sum': 'total_out_amount', 'txn_amt_count': 'total_out_txns', 
    'txn_amt_mean': 'avg_out_amount', 'txn_amt_std': 'std_out_amount', 
    'txn_amt_max': 'max_out_amount', 'to_acct_nunique': 'unique_to_accts', 
    'day_idx_nunique': 'unique_out_txn_days', 'is_late_night_sum': 'late_night_out_txn_count', 
    'is_late_night_mean': 'late_night_out_txn_ratio', 
    'is_round_number_txn_mean': 'round_number_out_txn_ratio',
    'is_registered_txn_sum': 'registered_out_txn_count', # [NEW]
    'is_registered_txn_mean': 'registered_out_txn_ratio' # [NEW]
}).drop(columns=['day_idx_min', 'day_idx_max'])

# --- Vectorized In-degree Features ---
print("Calculating in-degree features (Vectorized)...")
in_features = df_trans.groupby('to_acct').agg({
    'txn_amt': ['sum', 'mean', 'count'], 
    'from_acct': ['nunique']
})
in_features.columns = ['in_' + '_'.join(col).strip() for col in in_features.columns.values]
in_features = in_features.rename(columns={
    'in_txn_amt_sum': 'total_in_amount', 'in_txn_amt_count': 'total_in_txns', 
    'in_txn_amt_mean': 'avg_in_amount', 'in_from_acct_nunique': 'unique_from_accts'
})

# --- Advanced Temporal Features (Linear Regression Slope) ---
print("Extracting advanced temporal features (Vectorized Math)...")

# Calculate Slope using closed-form linear regression formula:
# Slope = (N * Sum(XY) - Sum(X)*Sum(Y)) / (N * Sum(X^2) - Sum(X)^2)
df_trans['xy'] = df_trans['day_idx'] * df_trans['txn_amt']
df_trans['xx'] = df_trans['day_idx'] ** 2

trend_stats = df_trans.groupby('from_acct').agg({
    'txn_amt': ['sum', 'count'],
    'day_idx': ['sum'],
    'xy': ['sum'],
    'xx': ['sum']
})
trend_stats.columns = ['y_sum', 'n', 'x_sum', 'xy_sum', 'xx_sum']

epsilon = 1e-9
numerator = (trend_stats['n'] * trend_stats['xy_sum']) - (trend_stats['x_sum'] * trend_stats['y_sum'])
denominator = (trend_stats['n'] * trend_stats['xx_sum']) - (trend_stats['x_sum'] ** 2)
trend_stats['amt_trend'] = numerator / (denominator + epsilon)

# Activity Density
span = out_features['account_lifespan_days'] + 1
trend_stats['activity_density'] = out_features['unique_out_txn_days'] / span

# Volatility (Coefficient of Variation)
trend_stats['amt_volatility'] = out_features['std_out_amount'] / (out_features['avg_out_amount'] + epsilon)

adv_temporal_features = trend_stats[['amt_trend', 'activity_density', 'amt_volatility']].fillna(0)
del trend_stats, df_trans['xy'], df_trans['xx'] # Memory cleanup

# --- Behavioral Anomaly Z-Scores ---
print("Calculating behavioral anomaly Z-scores (Vectorized)...")
# Merge mean/std back to transactions to calculate deviation for every single transaction
temp_stats = out_features[['avg_out_amount', 'std_out_amount']]
df_trans = df_trans.merge(temp_stats, left_on='from_acct', right_index=True, how='left')
df_trans['txn_zscore'] = (df_trans['txn_amt'] - df_trans['avg_out_amount']) / (df_trans['std_out_amount'].fillna(0) + epsilon)
max_zscore_features = df_trans.groupby('from_acct')['txn_zscore'].max().to_frame(name='max_txn_zscore')

# --- Sparse PageRank Calculation ---
print("Calculating PageRank (Sparse Matrix Method)...")
# Map all unique accounts to integers 0..N
all_accts = pd.unique(np.concatenate([df_trans['from_acct'].astype(str), df_trans['to_acct'].astype(str)]))
acct_map = {acct: i for i, acct in enumerate(all_accts)}
n_nodes = len(all_accts)

# Build Sparse Matrix (COO format)
from_idx = df_trans['from_acct'].map(acct_map).values
to_idx = df_trans['to_acct'].map(acct_map).values
data = np.ones(len(from_idx))
A = sparse.coo_matrix((data, (from_idx, to_idx)), shape=(n_nodes, n_nodes), dtype=np.float32)

pr_scores = fast_pagerank(A)
pagerank_features = pd.DataFrame(pr_scores, index=all_accts, columns=['pagerank'])

# --- Feature Integration ---
print("Integrating all features...")
all_accts_df = pd.DataFrame(index=all_accts)
node_features = all_accts_df.join(out_features)\
                            .join(in_features)\
                            .join(adv_temporal_features)\
                            .join(max_zscore_features)\
                            .join(pagerank_features)\
                            .join(reg_static_out)\
                            .join(reg_static_in) # [NEW] Join static register features

node_features.fillna(0, inplace=True)

# --- Interaction & Ratio Features ---
print("Adding interaction features...")
node_features['net_flow'] = node_features['total_in_amount'] - node_features['total_out_amount']
node_features['in_out_amount_ratio'] = node_features['total_in_amount'] / (node_features['total_out_amount'] + epsilon)
node_features['hub_score'] = node_features['unique_from_accts'] * node_features['unique_to_accts']
node_features['risk_weighted_volume'] = node_features['late_night_out_txn_ratio'] * node_features['total_out_amount']
node_features['volatility_weighted_volume'] = node_features['amt_volatility'] * node_features['total_out_amount']
node_features['pagerank_weighted_flow'] = node_features['pagerank'] * node_features['net_flow']
node_features['avg_daily_out_txns'] = node_features['total_out_txns'] / (node_features['account_lifespan_days'] + epsilon)
node_features['avg_daily_out_amount'] = node_features['total_out_amount'] / (node_features['account_lifespan_days'] + epsilon)
node_features['avg_daily_in_txns'] = node_features['total_in_txns'] / (node_features['account_lifespan_days'] + epsilon)
node_features['anomaly_magnitude'] = node_features['max_txn_zscore'] * node_features['avg_out_amount']

# [NEW] Registered interaction features
# Calculate "Unregistered" (Riskier) volume and counts
node_features['unregistered_out_count'] = node_features['total_out_txns'] - node_features['registered_out_txn_count']
node_features['unregistered_ratio'] = 1.0 - node_features['registered_out_txn_ratio']
# High volatility combined with high unregistered ratio is suspicious
node_features['risk_unregistered_volatility'] = node_features['unregistered_ratio'] * node_features['amt_volatility']

# Final Cleanup
node_features.replace([np.inf, -np.inf], 0, inplace=True)
node_features.fillna(0, inplace=True)
print(f"✅ Final feature count: {node_features.shape[1]}") 

# Garbage collection to free RAM before scaling
del df_trans, out_features, in_features, adv_temporal_features, max_zscore_features, pagerank_features, A
gc.collect()

# --- 3. Feature Scaling ---
print("\nStep 3: Scaling node features...")
scaler = StandardScaler()
# Convert to float32 to reduce memory footprint
features_val = node_features.values.astype(np.float32)
scaled_features_np = scaler.fit_transform(features_val)
node_features_scaled = pd.DataFrame(scaled_features_np, index=node_features.index, columns=node_features.columns)

# --- 4. Saving Artifacts ---
print("\nStep 4: Saving processed features to file...")
save_path = 'processed_features_with_interactions.pt' 
acct_to_idx = {acct_id: i for i, acct_id in enumerate(node_features_scaled.index)}
feature_tensor = torch.tensor(node_features_scaled.values, dtype=torch.float)

torch.save({
    'feature_tensor': feature_tensor, 
    'acct_to_idx': acct_to_idx, 
    'feature_names': list(node_features.columns),
    'scaler': scaler
}, save_path)
print(f"✅ Features saved to '{save_path}'")
print("\n--- Completed Successfully ---")
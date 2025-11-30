"""
Trains and evaluates a GraphSAGE model for fraud detection.

This script constitutes Part 2 of the pipeline, focusing on model training.
It performs the following steps:
1.  Loads node features (from Part 1) and raw data for graph/labels.
2.  Prepares a PyTorch Geometric (PyG) `Data` object, including graph
    structure, features, labels, and train/val/test splits.
3.  Defines a 3-layer GraphSAGE model with batch normalization and dropout.
4.  Implements training and testing functions, using a dampened weighted
    CrossEntropyLoss to handle class imbalance.
5.  Executes the training loop, tracking validation F1-score for
    early stopping and model checkpointing.
6.  Loads the best-performing model and evaluates it on the hold-out
    test set.
7.  Generates and displays plots of training metrics (loss, AUC, F1, etc.).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, confusion_matrix
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

### --- 0. Setup and Data Loading ---
print("--- Part 2: Model Training ---")
print("Step 0: Setup and Data Loading...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

try:
    # Load pre-processed features from Part 1
    saved_data = torch.load('processed_features_with_interactions.pt', map_location=device, weights_only=False)
    feature_tensor = saved_data['feature_tensor']
    acct_to_idx = saved_data['acct_to_idx']
    print("Pre-processed features loaded successfully.")
except FileNotFoundError:
    print("Error: 'processed_features_with_interactions.pt' not found. Please run Part 1 first.")
    exit()

try:
    # Load raw data for graph construction and labels
    df_trans = pd.read_csv("data/acct_transaction.csv", dtype={'from_acct': str, 'to_acct': str})
    df_alert = pd.read_csv("data/acct_alert.csv", dtype={'acct': str})
    print("Raw transaction and alert data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading raw data files: {e}")
    exit()

### --- 1. Prepare Graph Data and Splits ---
print("\nStep 1: Preparing Graph Data and Splits...")
# Filter transactions where both accounts are in our feature map
valid_trans = df_trans[df_trans['from_acct'].isin(acct_to_idx) & df_trans['to_acct'].isin(acct_to_idx)]
source_nodes = valid_trans['from_acct'].map(acct_to_idx).values
target_nodes = valid_trans['to_acct'].map(acct_to_idx).values

# Create edge index and make the graph undirected
edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
edge_index = to_undirected(edge_index)
num_nodes = len(acct_to_idx)

# Create labels (y) tensor
y = torch.zeros(num_nodes, dtype=torch.long)
alert_indices_list = [acct_to_idx[acct] for acct in df_alert['acct'] if acct in acct_to_idx]
y[alert_indices_list] = 1

# Create stratified train/validation/test splits
indices = torch.arange(num_nodes)
train_idx, temp_idx, y_train, y_temp = train_test_split(indices, y, train_size=0.7, stratify=y, random_state=42)
val_idx, test_idx, _, _ = train_test_split(temp_idx, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Create boolean masks
train_mask = torch.zeros(num_nodes, dtype=torch.bool); train_mask[train_idx] = True
val_mask = torch.zeros(num_nodes, dtype=torch.bool); val_mask[val_idx] = True
test_mask = torch.zeros(num_nodes, dtype=torch.bool); test_mask[test_idx] = True

# Create the PyG Data object
graph_data = Data(x=feature_tensor, edge_index=edge_index, y=y,
                  train_mask=train_mask, val_mask=val_mask, test_mask=test_mask).to(device)
print("Graph data object created.")

### --- 2. GNN Model Definition ---
class GraphSAGE(nn.Module):
    """
    A 3-layer GraphSAGE model with Batch Normalization and Dropout.

    This model implements a stack of SAGEConv layers for node classification.
    The architecture is:
    SAGEConv -> BatchNorm -> ReLU -> Dropout
    SAGEConv -> BatchNorm -> ReLU -> Dropout
    SAGEConv (linear output)

    Args:
        in_channels (int): Dimensionality of the input node features.
        hidden_channels (int): Dimensionality of the hidden embeddings.
        out_channels (int): Dimensionality of the output (number of classes).
        dropout (float): Dropout probability.

    Attributes:
        conv1 (SAGEConv): The first graph convolutional layer.
        bn1 (nn.BatchNorm1d): Batch normalization for the first hidden layer.
        conv2 (SAGEConv): The second graph convolutional layer.
        bn2 (nn.BatchNorm1d): Batch normalization for the second hidden layer.
        conv3 (SAGEConv): The output graph convolutional layer.
        dropout (nn.Dropout): Dropout layer.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        """Initializes the GraphSAGE model layers."""
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        """
        Defines the forward pass of the GraphSAGE model.

        Args:
            x (torch.Tensor): The input node feature tensor.
            edge_index (torch.Tensor): The graph's edge index.

        Returns:
            torch.Tensor: The raw output logits for each node.
        """
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        return x

### --- 3. Training and Evaluation Flow ---
print("\nStep 3: Training and Evaluation...")
def train(model, data, optimizer, loss_fn):
    """
    Performs a single training step on the graph.

    Args:
        model (nn.Module): The GNN model to train.
        data (Data): The PyG graph data object.
        optimizer (torch.optim.Optimizer): The optimizer.
        loss_fn (torch.nn.Module): The loss function.

    Returns:
        float: The training loss for this step.
    """
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(model, data, mask):
    """
    Evaluates the model on a specified node mask (e.g., validation or test).

    This function operates in `torch.no_grad()` mode.

    Args:
        model (nn.Module): The GNN model to evaluate.
        data (Data): The PyG graph data object.
        mask (torch.Tensor): The boolean mask (e.g., `data.val_mask` or
                             `data.test_mask`) specifying which nodes
                             to evaluate.

    Returns:
        tuple: A tuple containing (auc, recall, precision, f1) scores.
    """
    model.eval()
    out = model(data.x, data.edge_index)
    # Get probabilities for the positive class (class 1)
    pred_proba = F.softmax(out[mask], dim=1)[:, 1]
    # Get predicted classes (0 or 1) based on argmax
    pred_class = out[mask].argmax(dim=1)
    y_true = data.y[mask]
    
    # Calculate metrics
    auc = roc_auc_score(y_true.cpu(), pred_proba.cpu())
    recall = recall_score(y_true.cpu(), pred_class.cpu(), zero_division=0)
    precision = precision_score(y_true.cpu(), pred_class.cpu(), zero_division=0)
    f1 = f1_score(y_true.cpu(), pred_class.cpu(), zero_division=0)
    return auc, recall, precision, f1

# Calculate dampened class weights for imbalance
num_positives = graph_data.y[graph_data.train_mask].sum().item()
num_negatives = graph_data.train_mask.sum().item() - num_positives
pos_weight_ratio = num_negatives / num_positives
# Use sqrt to dampen the effect of the large weight
dampened_weight = np.sqrt(pos_weight_ratio)
class_weights = torch.tensor([1.0, dampened_weight], dtype=torch.float).to(device)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
print(f"Original weight ratio: {pos_weight_ratio:.2f}")
print(f"Dampened positive class weight (sqrt): {dampened_weight:.2f}")


# --- Hyperparameters and Initialization ---
HIDDEN_CHANNELS = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 50000
EARLY_STOPPING_PATIENCE = 10
model = GraphSAGE(in_channels=graph_data.num_node_features, hidden_channels=HIDDEN_CHANNELS, out_channels=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Initialize learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

# UPDATED: Initialize history with separate keys for Train and Val
history = {
    'epoch': [], 
    'train_loss': [], 'val_loss': [],
    'train_auc': [], 'val_auc': [],
    'train_recall': [], 'val_recall': [],
    'train_precision': [], 'val_precision': [],
    'train_f1': [], 'val_f1': []
}
best_val_f1 = -1
best_epoch = 0
patience_counter = 0

start_time = time.time()
for epoch in range(1, EPOCHS + 1):
    loss = train(model, graph_data, optimizer, loss_fn)
    
    # Validation and logging every 10 epochs
    if epoch % 10 == 0:
        # Evaluate on Validation set
        val_auc, val_recall, val_precision, val_f1 = test(model, graph_data, graph_data.val_mask)
        
        # UPDATED: Evaluate on Training set
        train_auc, train_recall, train_precision, train_f1 = test(model, graph_data, graph_data.train_mask)
        
        # UPDATED: Calculate Validation Loss explicitly
        model.eval()
        with torch.no_grad():
            out = model(graph_data.x, graph_data.edge_index)
            val_loss = loss_fn(out[graph_data.val_mask], graph_data.y[graph_data.val_mask]).item()

        print(f'Epoch: {epoch:04d}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}')
        
        # Store metrics
        history['epoch'].append(epoch)
        history['train_loss'].append(loss)
        history['val_loss'].append(val_loss)
        
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)
        
        history['train_recall'].append(train_recall)
        history['val_recall'].append(val_recall)
        
        history['train_precision'].append(train_precision)
        history['val_precision'].append(val_precision)
        
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        
        # Adjust learning rate based on validation F1
        scheduler.step(val_f1)
        
        # Early stopping logic
        current_metric = val_f1
        if current_metric > best_val_f1:
            best_val_f1 = current_metric
            best_epoch = epoch
            torch.save(model.state_dict(), 'best_model.pt')
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n--- Early stopping triggered at epoch {epoch} ---")
            break

print(f"\n--- Training Finished ---")
print(f"Total time: {(time.time() - start_time)/60:.2f} minutes")
print(f"Best validation F1-score: {best_val_f1:.4f} at epoch {best_epoch}")

### --- 4. Final Evaluation ---
print("\nStep 4: Evaluating on Test Set...")
# Load the best model saved during training
model.load_state_dict(torch.load('best_model.pt'))

# Evaluate on the test set using the default argmax (0.5) threshold
final_auc, final_recall, final_precision, final_f1 = test(model, graph_data, graph_data.test_mask)

print(f"\n--- Test Set Performance (using default argmax/0.5 threshold) ---")
print(f"AUC: {final_auc:.4f}")
print(f"Recall: {final_recall:.4f}")
print(f"Precision: {final_precision:.4f}")
print(f"F1-Score: {final_f1:.4f}")


### --- 5. Visualization ---
print("\nStep 5: Visualizing Training Process (Train vs Val)...")
fig, ax = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Training and Validation Metrics Comparison')

# Plot 1: Loss
ax[0, 0].plot(history['epoch'], history['train_loss'], label='Training Loss')
ax[0, 0].plot(history['epoch'], history['val_loss'], label='Validation Loss')
ax[0, 0].set_title('Loss')
ax[0, 0].legend()
ax[0, 0].grid(True)

# Plot 2: AUC
ax[0, 1].plot(history['epoch'], history['train_auc'], label='Training AUC', color='blue')
ax[0, 1].plot(history['epoch'], history['val_auc'], label='Validation AUC', color='orange')
ax[0, 1].set_title('AUC')
ax[0, 1].legend()
ax[0, 1].grid(True)

# Plot 3: Recall & Precision
# Solid line for Training, Dashed for Validation
ax[1, 0].plot(history['epoch'], history['train_recall'], label='Train Recall', color='green', linestyle='-')
ax[1, 0].plot(history['epoch'], history['val_recall'], label='Val Recall', color='green', linestyle='--')
ax[1, 0].plot(history['epoch'], history['train_precision'], label='Train Precision', color='red', linestyle='-')
ax[1, 0].plot(history['epoch'], history['val_precision'], label='Val Precision', color='red', linestyle='--')
ax[1, 0].set_title('Recall & Precision')
ax[1, 0].legend()
ax[1, 0].grid(True)

# Plot 4: F1-Score
ax[1, 1].plot(history['epoch'], history['train_f1'], label='Training F1', color='purple', linestyle='-')
ax[1, 1].plot(history['epoch'], history['val_f1'], label='Validation F1', color='purple', linestyle='--')
ax[1, 1].set_title('F1-Score')
ax[1, 1].legend()
ax[1, 1].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
print("\n--- Part 2 Completed ---")
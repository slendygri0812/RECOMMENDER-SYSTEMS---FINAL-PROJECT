import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import tensorflow as tf
from data_utils import load_and_preprocess_data, get_item_descriptions, encode_ids, split_and_generate_sequences, get_tf_datasets

csv_path = "steam_reviews_bruteforce.csv"

# Let's run a test of the workflow
df_filtered = load_and_preprocess_data(csv_path, min_interactions=5)

# Verify sequence counts and timestamps
print("\n--- PASO 1 TensorFlow Verification ---")
num_users_filtered = df_filtered['steamid'].nunique()
num_items_filtered = df_filtered['appid'].nunique()
num_interactions_filtered = len(df_filtered)
sparsity_filtered = 1.0 - (num_interactions_filtered / (num_users_filtered * num_items_filtered))

print(f"Number of rows after filtering: {num_interactions_filtered}")
print(f"Number of unique users: {num_users_filtered}")
print(f"Number of unique games: {num_items_filtered}")
print(f"Dataset Density: {((1.0 - sparsity_filtered) * 100):.4f}%")
print(f"Dataset Sparsity: {(sparsity_filtered * 100):.4f}%")

# Try a sample user sequence
sample_user = df_filtered['steamid'].iloc[0]
user_data = df_filtered[df_filtered['steamid'] == sample_user]
print(f"\nSample User: {sample_user}")
print(f"Interaction Timestamps (Should be sorted):")
for idx, row in user_data.iterrows():
    print(f"  Game: {row['appid']}, Time: {row['timestamp_created']}")

# 2. Get Item Descriptions
item_descriptions = get_item_descriptions(df_filtered)
sample_appid = list(item_descriptions.keys())[0]
print(f"\nSample Game: {sample_appid}")
print(f"Description length: {len(item_descriptions[sample_appid].split())} words")
print(f"Description: '{item_descriptions[sample_appid][:200]}...'")

# 3. ID Encoding
df_filtered, user_to_id, item_to_id, id_to_description = encode_ids(df_filtered, item_descriptions)
print(f"\nUser sequential mapping size: {len(user_to_id)}")
print(f"Item sequential mapping size: {len(item_to_id)}")
print(f"Padding description at ID 0 exists: {0 in id_to_description}")

# 4. Sequence splitting and padding
train_seqs, train_targets, val_seqs, val_targets, test_seqs, test_targets = split_and_generate_sequences(df_filtered, max_len=10)

print(f"\nSplit statistics:")
print(f"  Train inputs shape: {train_seqs.shape}, Targets shape: {train_targets.shape}")
print(f"  Val inputs shape: {val_seqs.shape}, Targets shape: {val_targets.shape}")
print(f"  Test inputs shape: {test_seqs.shape}, Targets shape: {test_targets.shape}")

# Inspect first sample user sequence
print(f"\nFirst User training input sequence: {train_seqs[0]}")
print(f"First User training target sequence: {train_targets[0]}")
print(f"First User validation input sequence: {val_seqs[0]} -> Target: {val_targets[0]}")
print(f"First User test input sequence: {test_seqs[0]} -> Target: {test_targets[0]}")

train_dataset, val_dataset, test_dataset = get_tf_datasets(
    train_seqs, train_targets, val_seqs, val_targets, test_seqs, test_targets, batch_size=32
)

# Fetch first batch
for seqs, targets in train_dataset:
    print(f"\nTensorFlow Dataset Verification:")
    print(f"  Train Batch input shape: {seqs.shape}")
    print(f"  Train Batch target shape: {targets.shape}")
    break

print("\nPASO 1 TensorFlow logic successfully verified!")

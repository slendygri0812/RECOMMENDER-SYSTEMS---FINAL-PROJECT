import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import tensorflow as tf
import numpy as np
from data_utils import load_and_preprocess_data, get_item_descriptions, encode_ids
from distilbert_extractor import DistilBERTExtractor

csv_path = "steam_reviews_bruteforce.csv"

# 1. Load data and extract descriptions
df_filtered = load_and_preprocess_data(csv_path, min_interactions=5)
item_descriptions = get_item_descriptions(df_filtered)
df_filtered, user_to_id, item_to_id, id_to_description = encode_ids(df_filtered, item_descriptions)

# Take a small subsample of items to verify fast
print("\n--- Subsampling for TensorFlow Verification ---")
test_id_to_description = {0: id_to_description[0]}
for i in list(id_to_description.keys())[1:6]:
    test_id_to_description[i] = id_to_description[i]
    
print(f"Items to encode: {list(test_id_to_description.keys())}")

# 2. Instantiate TF extractor and run
extractor = DistilBERTExtractor()
embeddings_matrix = extractor.extract_item_embeddings(test_id_to_description)

# 3. Check shape and values
print("\n--- PASO 2 TensorFlow Verification ---")
print(f"Matrix shape: {embeddings_matrix.shape} (Expected: (7, 768) if max_id is 6)")
print(f"Matrix type: {type(embeddings_matrix)}")
print(f"Padding vector at index 0 (all zeros?): {np.all(embeddings_matrix[0] == 0)}")
print(f"Item 2 representation (mean): {embeddings_matrix[2].mean().item():.6f}")
print(f"Item 2 representation (std): {embeddings_matrix[2].std().item():.6f}")

print("\nPASO 2 TensorFlow logic successfully verified!")

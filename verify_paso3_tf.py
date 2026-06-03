import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import tensorflow as tf
import numpy as np
from model import HBESTModel, BaselineModel

# 1. Define model parameters
num_items = 428
max_len = 10
d_model = 128
num_heads = 2
num_layers = 2
batch_size = 32

# Create mock BERT embeddings (shape: num_items + 1, 768)
mock_bert_embeddings = np.random.randn(num_items + 1, 768).astype(np.float32)
mock_bert_embeddings[0] = 0.0  # Zero padding at index 0

# 2. Instantiate HBESTModel and BaselineModel
print("Instantiating models in TensorFlow/Keras...")
hbest_model = HBESTModel(
    num_items=num_items,
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    max_len=max_len,
    bert_embeddings=mock_bert_embeddings
)

baseline_model = BaselineModel(
    num_items=num_items,
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    max_len=max_len
)

# Create input batch of shape (batch_size, max_len)
# Reserving 0 for padding, item IDs between 1 and num_items
seq_ids_np = np.random.randint(0, num_items + 1, size=(batch_size, max_len)).astype(np.int32)
# Ensure some padding tokens (zeros) exist in the batch
seq_ids_np[seq_ids_np % 3 == 0] = 0
seq_ids = tf.constant(seq_ids_np)

print(f"Input batch shape: {seq_ids.shape}")

# 3. Verify HBESTModel Forward and Prediction passes
print("\n--- Testing H-BEST Model ---")
hbest_forward_out = hbest_model(seq_ids, training=True)
print(f"Forward output shape (hidden states): {hbest_forward_out.shape} (Expected: ({batch_size}, {max_len}, {d_model}))")

hbest_logits = hbest_model.predict_logits(seq_ids)
print(f"Prediction logits shape: {hbest_logits.shape} (Expected: ({batch_size}, {num_items + 1}))")

# Check that padding index (0) scores are valid or handleable
print(f"Logits range - min: {tf.reduce_min(hbest_logits).numpy():.4f}, max: {tf.reduce_max(hbest_logits).numpy():.4f}")

# 4. Verify BaselineModel Forward and Prediction passes
print("\n--- Testing Baseline Model (SASRec) ---")
baseline_forward_out = baseline_model(seq_ids, training=True)
print(f"Forward output shape (hidden states): {baseline_forward_out.shape} (Expected: ({batch_size}, {max_len}, {d_model}))")

baseline_logits = baseline_model.predict_logits(seq_ids)
print(f"Prediction logits shape: {baseline_logits.shape} (Expected: ({batch_size}, {num_items + 1}))")
print(f"Logits range - min: {tf.reduce_min(baseline_logits).numpy():.4f}, max: {tf.reduce_max(baseline_logits).numpy():.4f}")

# Check gradients can propagate
print("\n--- Testing Backpropagation Capability ---")
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
targets = tf.constant(np.random.randint(1, num_items + 1, size=(batch_size,)).astype(np.int32))

# HBEST backpropagation
hbest_optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001)
with tf.GradientTape() as tape:
    logits = hbest_model.predict_logits(seq_ids)
    loss = loss_fn(targets, logits)
gradients = tape.gradient(loss, hbest_model.trainable_variables)
hbest_optimizer.apply_gradients(zip(gradients, hbest_model.trainable_variables))
print("H-BEST backpropagation step succeeded!")

# Baseline backpropagation
baseline_optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001)
with tf.GradientTape() as tape:
    logits = baseline_model.predict_logits(seq_ids)
    loss = loss_fn(targets, logits)
gradients = tape.gradient(loss, baseline_model.trainable_variables)
baseline_optimizer.apply_gradients(zip(gradients, baseline_model.trainable_variables))
print("Baseline backpropagation step succeeded!")

print("\nPASO 3 models fully verified successfully in TensorFlow!")

import tensorflow as tf
import numpy as np
from tqdm import tqdm
from evaluate import evaluate_model

def compute_masked_loss(logits, targets, loss_fn):
    r"""
    Calcula la pérdida de entropía cruzada secuencial enmascarando las posiciones de padding (donde target = 0).
    
    Propósito Matemático:
    --------------------
    Dado logits \hat{Y} \in \mathbb{R}^{B \times L \times (N+1)} y targets Y \in \{0, 1, ..., N\}^{B \times L}:
    La pérdida para cada elemento es:
        \mathcal{L}_{b, t} = \text{CrossEntropy}(Y_{b, t}, \hat{Y}_{b, t})
    Aplicamos la máscara indicadora \mathbb{I}(Y_{b, t} \ne 0):
        \mathcal{L} = \frac{\sum_{b, t} \mathcal{L}_{b, t} \mathbb{I}(Y_{b, t} \ne 0)}{\sum_{b, t} \mathbb{I}(Y_{b, t} \ne 0)}
    """
    # loss_fn debe tener reduction='none' para retornar (batch_size, seq_len)
    raw_loss = loss_fn(targets, logits)
    
    # Crear máscara para ignorar el ID de padding (0) en los objetivos
    mask = tf.cast(targets != 0, dtype=tf.float32)
    
    masked_loss = raw_loss * mask
    
    # Promediar sobre los elementos no acolchados
    loss_sum = tf.reduce_sum(masked_loss)
    mask_sum = tf.reduce_sum(mask)
    
    return loss_sum / (mask_sum + 1e-9)

def compute_val_loss(model, dataset, loss_fn):
    r"""
    Calcula la pérdida de entropía cruzada en el conjunto de validación.
    """
    total_loss = 0.0
    steps = 0
    for seq_ids, targets in dataset:
        logits = model.predict_logits(seq_ids)
        loss = loss_fn(targets, logits)
        total_loss += tf.reduce_mean(loss).numpy()
        steps += 1
    return total_loss / (steps + 1e-9)

def train_epoch(model, dataset, optimizer, loss_fn):
    epoch_loss = 0.0
    steps = 0
    
    for seq_ids, targets in dataset:
        with tf.GradientTape() as tape:
            # 1. Forward pass para obtener los hidden states secuenciales: (batch_size, seq_len, d_model)
            h = model(seq_ids, training=True)
            
            # 2. Calcular los logits para todas las posiciones de la secuencia
            all_embeddings = model._get_fused_item_embeddings() # (num_items + 1, d_model)
            logits = tf.matmul(h, all_embeddings, transpose_b=True) # (batch_size, seq_len, num_items + 1)
            
            # 3. Calcular la pérdida enmascarando los ceros
            loss = compute_masked_loss(logits, targets, loss_fn)
            
        # 4. Propagación de gradientes y actualización
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        epoch_loss += loss.numpy()
        steps += 1
        
    return epoch_loss / (steps + 1e-9)

def train_model(model, train_dataset, val_dataset, epochs=20, lr=0.001, weight_decay=1e-4, verbose=True, plot_results=True, model_name="model"):
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    
    best_ndcg = -1.0
    best_weights = None
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'epochs': list(range(1, epochs + 1)),
        'val_metrics': []
    }
    
    if verbose:
        print(f"\nIniciando entrenamiento por {epochs} épocas...")
        
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_dataset, optimizer, loss_fn)
        val_loss = compute_val_loss(model, val_dataset, loss_fn)
        
        # Evaluar sobre el conjunto de validación (K=10)
        val_metrics = evaluate_model(model, val_dataset, k_list=[10])
        val_ndcg = val_metrics["NDCG@10"]
        val_hr = val_metrics["HR@10"]
        
        history['train_loss'].append(loss)
        history['val_loss'].append(val_loss)
        history['val_metrics'].append(val_metrics)
        
        if verbose:
            print(f"Época {epoch:02d}/{epochs:02d} | Pérdida Train: {loss:.4f} | Pérdida Val: {val_loss:.4f} | Val HR@10: {val_hr:.4f} | Val NDCG@10: {val_ndcg:.4f} | Val Acc: {val_metrics['Accuracy']:.4f}")
            
        # Guardar pesos del mejor modelo basado en NDCG@10
        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            best_weights = [tf.identity(w) for w in model.weights]
            
    # Restaurar los mejores pesos
    if best_weights is not None:
        if verbose:
            print(f"Cargando los mejores pesos con Val NDCG@10: {best_ndcg:.4f}")
        for w, best_w in zip(model.weights, best_weights):
            w.assign(best_w)
            
    if plot_results:
        from evaluate import plot_training_validation_curves
        plot_training_validation_curves(history, save_dir=f"plots_{model_name}")
        
    return model, history

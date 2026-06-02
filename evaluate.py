import numpy as np
import tensorflow as tf

def evaluate_model(model, dataset, k_list=[5, 10]):
    r"""
    Evalúa el modelo sobre un tf.data.Dataset secuencial de validación o prueba.
    
    Propósito Matemático:
    --------------------
    Para cada secuencia de usuario S_u, el modelo predice los logits sobre los ítems candidatos.
    Encontramos el ranking de la predicción respecto al ítem objetivo real g_u.
    
    1. Hit Rate @ K (HR@K):
       $$HR@K = \frac{1}{|U|} \sum_{u \in U} \mathbb{I}(\text{rank}(g_u) \le K)$$
       donde \mathbb{I} es la función indicadora.
       
    2. Normalized Discounted Cumulative Gain @ K (NDCG@K):
       $$NDCG@K = \frac{1}{|U|} \sum_{u \in U} \frac{\mathbb{I}(\text{rank}(g_u) \le K)}{\log_2(\text{rank}(g_u) + 1)}$$
    """
    # Inicializar sumadores de métricas
    metrics = {f"HR@{k}": 0.0 for k in k_list}
    metrics.update({f"NDCG@{k}": 0.0 for k in k_list})
    total_users = 0
    
    for seq_ids, targets in dataset:
        # Generar las predicciones logits: (batch_size, num_items + 1)
        logits = model.predict_logits(seq_ids)
        
        # Ocultar el ID 0 (padding) asignándole un valor extremadamente bajo
        # para que nunca sea recomendado en el Top-K
        num_candidates = tf.shape(logits)[1]
        mask_padding = tf.one_hot(0, depth=num_candidates, on_value=-1e9, off_value=0.0)
        logits = logits + mask_padding
        
        # Convertir a numpy para procesamiento métrico rápido
        logits_np = logits.numpy()
        targets_np = targets.numpy()
        
        batch_size = len(targets_np)
        total_users += batch_size
        
        # Obtener los índices ordenados de mayor a menor probabilidad
        top_indices = np.argsort(-logits_np, axis=1)
        
        for b in range(batch_size):
            target_item = targets_np[b]
            user_ranking = top_indices[b]
            
            # Encontrar en qué rango quedó el ítem real (1-based rank)
            ranks = np.where(user_ranking == target_item)[0]
            if len(ranks) > 0:
                rank = ranks[0] + 1  # Rango es 1-based index (1, 2, ...)
            else:
                rank = 999999  # Fuera de rango
                
            for k in k_list:
                if rank <= k:
                    metrics[f"HR@{k}"] += 1.0
                    metrics[f"NDCG@{k}"] += 1.0 / np.log2(rank + 1)
                    
    # Promediar métricas sobre el total de usuarios
    if total_users > 0:
        for key in metrics:
            metrics[key] /= total_users
            
    return metrics

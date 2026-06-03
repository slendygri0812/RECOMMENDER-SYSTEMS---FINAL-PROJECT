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
    metrics.update({f"Precision@{k}": 0.0 for k in k_list})
    metrics.update({f"Recall@{k}": 0.0 for k in k_list})
    metrics["Accuracy"] = 0.0
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
                
            if rank == 1:
                metrics["Accuracy"] += 1.0
                
            for k in k_list:
                if rank <= k:
                    metrics[f"HR@{k}"] += 1.0
                    metrics[f"NDCG@{k}"] += 1.0 / np.log2(rank + 1)
                    metrics[f"Precision@{k}"] += 1.0 / k
                    metrics[f"Recall@{k}"] += 1.0
                    
    # Promediar métricas sobre el total de usuarios
    if total_users > 0:
        for key in metrics:
            metrics[key] /= total_users
            
    return metrics

def plot_training_validation_curves(history, save_dir="."):
    """
    Grafica las curvas de pérdida de entrenamiento vs validación y las curvas de métricas de validación por época.
    `history` debe contener:
      - 'train_loss': lista de floats (pérdida de entrenamiento por época)
      - 'val_loss': lista de floats (pérdida de validación por época)
      - 'epochs': lista de enteros con el índice de las épocas
      - 'val_metrics': lista de diccionarios con métricas de evaluación
    """
    import os
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Advertencia: No se pudo importar matplotlib.pyplot. Saltando graficación de curvas.")
        return
        
    os.makedirs(save_dir, exist_ok=True)
    epochs = history.get('epochs', list(range(1, len(history['train_loss']) + 1)))
    
    # 1. Graficar Curvas de Pérdida
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], label='Pérdida Entrenamiento', color='#FF6B6B', marker='o', linewidth=2)
    if 'val_loss' in history and len(history['val_loss']) > 0:
        plt.plot(epochs, history['val_loss'], label='Pérdida Validación', color='#4D96FF', marker='s', linewidth=2)
    plt.title('Curvas de Pérdida de Entrenamiento y Validación', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Pérdida (Entropía Cruzada)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    loss_path = os.path.join(save_dir, "loss_curves.png")
    plt.savefig(loss_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> Gráfica de pérdidas guardada en: '{loss_path}'")
    
    # 2. Graficar Curvas de Métricas de Validación
    if 'val_metrics' in history and len(history['val_metrics']) > 0:
        plt.figure(figsize=(12, 7))
        metrics_keys = history['val_metrics'][0].keys()
        
        # Seleccionar métricas clave a graficar (ej. Accuracy, HR@10, NDCG@10, Precision@10, Recall@10)
        selected_metrics = [k for k in metrics_keys if k in ['Accuracy', 'HR@10', 'NDCG@10', 'Precision@10', 'Recall@10']]
        colors = ['#FFD93D', '#6BCB77', '#4D96FF', '#FF6B6B', '#9B5DE5']
        
        for idx, metric_name in enumerate(selected_metrics):
            values = [m[metric_name] for m in history['val_metrics']]
            plt.plot(epochs, values, label=metric_name, color=colors[idx % len(colors)], marker='^', linewidth=2)
            
        plt.title('Métricas de Validación por Época', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Época', fontsize=12)
        plt.ylabel('Valor de la Métrica', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=11, loc='lower right')
        metrics_path = os.path.join(save_dir, "metrics_curves.png")
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f" -> Gráfica de métricas de validación guardada en: '{metrics_path}'")

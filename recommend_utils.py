import numpy as np
import tensorflow as tf

def get_user_recommendations(model, user_seq_ids, id_to_description, top_k=10, filter_seen=True, user_history_ids=None):
    """
    Genera recomendaciones de juegos para un usuario a partir de su historial de secuencias de IDs de ítems.
    
    Parámetros:
    -----------
    model : tf.keras.Model
        El modelo secuencial entrenado (H-BEST o Baseline).
    user_seq_ids : list o np.ndarray (1D)
        La secuencia histórica del usuario.
    id_to_description : dict
        Mapeo de ID secuencial de ítem a descripción textual.
    top_k : int
        Número de recomendaciones a generar.
    filter_seen : bool
        Si es True, se filtran los juegos que el usuario ya ha jugado/reseñado en su historial.
    user_history_ids : list o set (opcional)
        Colección de IDs para filtrar (si difiere de `user_seq_ids`). Por defecto usa `user_seq_ids`.
        
    Retorna:
    --------
    list of dicts
        Lista de juegos recomendados con sus IDs, puntuaciones (logits) y descripciones.
    """
    # Asegurar que el input sea un array de tamaño max_len
    max_len = model.max_len
    seq = list(user_seq_ids)
    
    # Pre-padear con ceros si es necesario
    if len(seq) < max_len:
        padded_seq = [0] * (max_len - len(seq)) + seq
    else:
        padded_seq = seq[-max_len:]
        
    # Obtener logits de predicción
    input_tensor = tf.constant([padded_seq], dtype=tf.int32)
    logits = model.predict_logits(input_tensor)[0]
    
    # Convertir logits a numpy
    logits_np = logits.numpy().copy()
    
    # Enmascarar el ID de padding (0)
    logits_np[0] = -1e9
    
    # Enmascarar ítems ya vistos
    if filter_seen:
        seen_set = set(user_history_ids) if user_history_ids is not None else set(seq)
        # Omitir el padding 0 de los vistos
        seen_set.discard(0)
        for item_id in seen_set:
            if item_id < len(logits_np):
                logits_np[item_id] = -1e9
                
    # Obtener el Top-K ordenado de mayor a menor logit
    top_indices = np.argsort(-logits_np)[:top_k]
    
    recommendations = []
    for idx in top_indices:
        score = logits_np[idx]
        if score <= -1e8:
            continue  # Todos los candidatos válidos fueron agotados
        recommendations.append({
            'seq_id': int(idx),
            'score': float(score),
            'description': id_to_description.get(idx, "Sin descripción disponible.")
        })
        
    return recommendations

def get_top_k_recommendations(model, batch_seq_ids, id_to_description, k=10, filter_seen=True):
    """
    Genera el listado Top-K de recomendaciones para un lote (batch) de secuencias de usuarios.
    
    Parámetros:
    -----------
    model : tf.keras.Model
        El modelo secuencial entrenado.
    batch_seq_ids : np.ndarray (forma: [batch_size, max_len])
        Lote de secuencias de entrada de usuarios.
    id_to_description : dict
        Mapeo de ID de juego a descripción.
    k : int
        Tamaño de la lista Top-K.
    filter_seen : bool
        Si es True, filtra los juegos vistos en cada secuencia del lote.
        
    Retorna:
    --------
    list of lists of dicts
        Para cada usuario en el lote, retorna su lista Top-K recomendada.
    """
    batch_logits = model.predict_logits(tf.constant(batch_seq_ids, dtype=tf.int32))
    
    # Enmascarar padding (ID 0)
    num_candidates = tf.shape(batch_logits)[1]
    mask_padding = tf.one_hot(0, depth=num_candidates, on_value=-1e9, off_value=0.0)
    batch_logits = batch_logits + mask_padding
    
    batch_logits_np = batch_logits.numpy()
    
    results = []
    for b in range(len(batch_seq_ids)):
        logits = batch_logits_np[b].copy()
        
        if filter_seen:
            seen_items = set(batch_seq_ids[b])
            seen_items.discard(0)
            for item_id in seen_items:
                if item_id < len(logits):
                    logits[item_id] = -1e9
                    
        top_indices = np.argsort(-logits)[:k]
        user_recs = []
        for idx in top_indices:
            score = logits[idx]
            if score <= -1e8:
                continue
            user_recs.append({
                'seq_id': int(idx),
                'score': float(score),
                'description': id_to_description.get(idx, "Sin descripción disponible.")
            })
        results.append(user_recs)
        
    return results

def get_similar_users(model, query_seq, all_users_seqs, top_n=5):
    """
    Encuentra los usuarios más similares a un usuario de consulta basado en sus representaciones latentes.
    La representación del usuario se obtiene de los hidden states finales del Transformer en H-BEST.
    
    Parámetros:
    -----------
    model : tf.keras.Model
        El modelo secuencial entrenado.
    query_seq : list o np.ndarray (1D)
        La secuencia histórica del usuario de consulta.
    all_users_seqs : np.ndarray (forma: [num_users, max_len])
        Secuencias históricas de todos los usuarios en el catálogo de comparación.
    top_n : int
        Número de usuarios similares a retornar.
        
    Retorna:
    --------
    list of dicts
        Lista de diccionarios con el índice del usuario similar y su score de similitud de coseno.
    """
    # 1. Obtener representación del usuario query
    max_len = model.max_len
    seq = list(query_seq)
    if len(seq) < max_len:
        padded_query = [0] * (max_len - len(seq)) + seq
    else:
        padded_query = seq[-max_len:]
        
    q_tensor = tf.constant([padded_query], dtype=tf.int32)
    q_rep = model(q_tensor, training=False)[:, -1, :].numpy()[0]  # [d_model]
    
    # 2. Obtener representaciones para todos los usuarios en batches
    representations = []
    dataset = tf.data.Dataset.from_tensor_slices(all_users_seqs).batch(256)
    for batch in dataset:
        h = model(batch, training=False)
        h_last = h[:, -1, :]
        representations.append(h_last.numpy())
    all_reps = np.vstack(representations)  # [num_users, d_model]
    
    # 3. Calcular similitud de coseno
    q_norm = np.linalg.norm(q_rep) + 1e-9
    all_norms = np.linalg.norm(all_reps, axis=1) + 1e-9
    
    # Similitud = (A . B) / (||A|| * ||B||)
    similarities = np.dot(all_reps, q_rep) / (all_norms * q_norm)
    
    # Obtener los más similares (excluyendo el usuario query si su similitud es exacta a 1.0)
    sorted_indices = np.argsort(-similarities)
    
    similar_users = []
    for idx in sorted_indices:
        sim = similarities[idx]
        # Guardar resultado
        similar_users.append({
            'user_index': int(idx),
            'similarity': float(sim)
        })
        if len(similar_users) >= top_n:
            break
            
    return similar_users

def get_similar_items(model, query_item_id, id_to_description, embedding_type='fused', top_n=5):
    """
    Encuentra los juegos más similares a un juego de consulta a partir de sus embeddings latentes.
    
    Parámetros:
    -----------
    model : tf.keras.Model
        El modelo entrenado.
    query_item_id : int
        El ID secuencial del juego de consulta.
    id_to_description : dict
        Mapeo de ID de juego a descripción.
    embedding_type : str
        El tipo de embedding a utilizar:
        - 'behavioral': Embeddings colaborativos del ID (`model.behavioral_emb`).
        - 'semantic': Embeddings proyectados desde DistilBERT (`model.semantic_projection` + BERT).
        - 'fused': La representación final sumada y real usada por H-BEST.
    top_n : int
        Número de ítems similares a retornar.
        
    Retorna:
    --------
    list of dicts
        Lista de juegos similares con sus IDs, descripciones y score de similitud de coseno.
    """
    # 1. Extraer matriz de embeddings según el tipo solicitado
    if embedding_type == 'behavioral':
        all_ids = tf.range(model.num_items + 1)
        embeddings = model.behavioral_emb(all_ids).numpy()
    elif embedding_type == 'semantic':
        if not getattr(model, 'use_semantic', False):
            raise ValueError("El modelo actual no tiene embeddings semánticos cargados.")
        all_ids = tf.range(model.num_items + 1)
        bert_features = tf.gather(model.bert_embeddings, all_ids)
        embeddings = model.semantic_projection(bert_features).numpy()
        # Forzar que el padding sea cero
        embeddings[0] = 0.0
    elif embedding_type == 'fused':
        embeddings = model._get_fused_item_embeddings().numpy()
    else:
        raise ValueError(f"Tipo de embedding no soportado: '{embedding_type}'. Elija entre 'behavioral', 'semantic' o 'fused'.")
        
    # Verificar que el query_item_id esté dentro del rango
    if query_item_id >= len(embeddings) or query_item_id <= 0:
        raise ValueError(f"El ID del juego '{query_item_id}' es inválido o es el padding.")
        
    query_emb = embeddings[query_item_id]
    
    # 2. Calcular similitud de coseno
    q_norm = np.linalg.norm(query_emb) + 1e-9
    all_norms = np.linalg.norm(embeddings, axis=1) + 1e-9
    
    similarities = np.dot(embeddings, query_emb) / (all_norms * q_norm)
    
    # Enmascarar el padding (índice 0) y el ítem query en sí mismo
    similarities[0] = -1.0
    similarities[query_item_id] = -1.0
    
    # Obtener el Top-N de mayor similitud
    top_indices = np.argsort(-similarities)[:top_n]
    
    similar_items = []
    for idx in top_indices:
        sim = similarities[idx]
        similar_items.append({
            'seq_id': int(idx),
            'similarity': float(sim),
            'description': id_to_description.get(idx, "Sin descripción disponible.")
        })
        
    return similar_items

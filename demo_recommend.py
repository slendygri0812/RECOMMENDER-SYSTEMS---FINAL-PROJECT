import os
import sys

# Reconfigurar salida estándar para soportar codificación UTF-8 en Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import tensorflow as tf

from data_utils import load_and_preprocess_data, get_item_descriptions, encode_ids, split_and_generate_sequences, get_tf_datasets
from distilbert_extractor import DistilBERTExtractor
from model import HBESTModel, BaselineModel
from train import train_model
from recommend_utils import get_user_recommendations, get_top_k_recommendations, get_similar_users, get_similar_items

def main():
    print("=========================================================================")
    print("        DEMO Y VERIFICACIÓN DE RECOMENDACIONES Y EMBEDDINGS (H-BEST)")
    print("=========================================================================")
    
    csv_path = "steam_reviews_bruteforce.csv"
    if not os.path.exists(csv_path):
        print(f"Error: No se encontró el dataset {csv_path} en el directorio actual.")
        return
        
    # --- Carga y preparación básica de datos ---
    print("\n[PASO 1] Carga y preparación rápida de datos...")
    df_filtered = load_and_preprocess_data(csv_path, min_interactions=5)
    item_descriptions = get_item_descriptions(df_filtered)
    df_filtered, user_to_id, item_to_id, id_to_description = encode_ids(df_filtered, item_descriptions)
    
    # Inverso de mapeo de IDs para mostrar información amigable
    id_to_item = {idx: appid for appid, idx in item_to_id.items()}
    
    max_len = 10
    (train_seqs, train_targets, 
     val_seqs, val_targets, 
     test_seqs, test_targets) = split_and_generate_sequences(df_filtered, max_len=max_len)
     
    train_dataset, val_dataset, test_dataset = get_tf_datasets(
        train_seqs, train_targets, 
        val_seqs, val_targets, 
        test_seqs, test_targets, 
        batch_size=64
    )
    
    # --- Carga de embeddings de BERT ---
    print("\n[PASO 2] Cargando embeddings semánticos...")
    cache_path = "distilbert_embeddings_cache.npy"
    if os.path.exists(cache_path):
        bert_embeddings = np.load(cache_path)
        print(f" -> Caché cargada con éxito. Forma: {bert_embeddings.shape}")
    else:
        print(" -> No se encontró caché. Ejecutando extractor...")
        extractor = DistilBERTExtractor()
        bert_embeddings = extractor.extract_item_embeddings(id_to_description)
        np.save(cache_path, bert_embeddings)
        
    # --- Inicialización y entrenamiento ultra rápido ---
    # Entrenaremos por solo 3 épocas para que la demo corra en segundos
    epochs = 3
    d_model = 64
    num_heads = 2
    num_layers = 1
    dropout = 0.2
    
    print(f"\n[PASO 3] Entrenando H-BEST por {epochs} épocas para demostración...")
    hbest_model = HBESTModel(
        num_items=len(item_to_id),
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        max_len=max_len,
        dropout=dropout,
        bert_embeddings=bert_embeddings
    )
    hbest_model, _ = train_model(
        model=hbest_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=epochs,
        lr=0.005,
        verbose=True,
        plot_results=False  # No sobreescribir plots finales
    )
    
    # --- DEMOSTRACIÓN DE FUNCIONALIDADES ---
    print("\n" + "="*70)
    print("                 DEMO DE FUNCIONES DE RECOMENDACIÓN")
    print("="*70)
    
    # Elegir un usuario aleatorio con interacciones interesantes
    query_user_idx = 10  # Usuario en la fila 10 del test set
    query_user_history = [int(i) for i in test_seqs[query_user_idx] if i > 0]
    original_user_id = list(user_to_id.keys())[list(user_to_id.values()).index(query_user_idx + 1)]
    
    print(f"\n>>> [FUNCIONALIDAD 1: RECOMENDACIONES DE USUARIOS ESPECÍFICOS]")
    print(f"ID del Usuario de Consulta (Original): {original_user_id} (Seq ID: {query_user_idx + 1})")
    print("Historial de juegos jugados por este usuario:")
    for item_idx in query_user_history:
        game_appid = id_to_item.get(item_idx)
        print(f"  - AppID: {game_appid:<10} | Juego: '{id_to_description.get(item_idx)[:60]}...'")
        
    # Generar recomendaciones con y sin filtro de ya jugados
    print("\nRecomendaciones generadas por H-BEST (Filtrando juegos del historial):")
    recs_filtered = get_user_recommendations(
        model=hbest_model,
        user_seq_ids=query_user_history,
        id_to_description=id_to_description,
        top_k=5,
        filter_seen=True
    )
    for idx, rec in enumerate(recs_filtered):
        rec_appid = id_to_item.get(rec['seq_id'])
        print(f"  {idx+1}. AppID: {rec_appid:<10} | Logit: {rec['score']:.4f} | Juego: '{rec['description'][:60]}...'")
        
    # 2. Top-K recommended list
    print(f"\n>>> [FUNCIONALIDAD 2: TOP-K RECOMMENDED LIST]")
    print("Generando lista Top-5 para un lote de 3 usuarios de prueba:")
    batch_seqs = test_seqs[15:18]
    batch_recs = get_top_k_recommendations(
        model=hbest_model,
        batch_seq_ids=batch_seqs,
        id_to_description=id_to_description,
        k=5,
        filter_seen=True
    )
    for u_idx, recs in enumerate(batch_recs):
        print(f"  Usuario de lote {u_idx + 1}:")
        for idx, rec in enumerate(recs):
            rec_appid = id_to_item.get(rec['seq_id'])
            print(f"    - Recomendación {idx+1}: AppID {rec_appid:<10} (Logit: {rec['score']:.2f}) -> '{rec['description'][:50]}...'")
            
    # 3. Similar Users
    print(f"\n>>> [FUNCIONALIDAD 3: SIMILAR USERS]")
    print(f"Buscando los 3 usuarios más similares al Usuario de Consulta:")
    similar_users = get_similar_users(
        model=hbest_model,
        query_seq=query_user_history,
        all_users_seqs=test_seqs,
        top_n=3
    )
    for idx, u_sim in enumerate(similar_users):
        u_seq_id = u_sim['user_index'] + 1
        orig_id = list(user_to_id.keys())[list(user_to_id.values()).index(u_seq_id)]
        print(f"  {idx+1}. Seq ID: {u_seq_id:<4} | Original ID: {orig_id} | Similitud de Coseno: {u_sim['similarity']:.4f}")
        
    # 4. Similar Items
    print(f"\n>>> [FUNCIONALIDAD 4: SIMILAR ITEMS POR EMBEDDINGS]")
    # Tomar el primer juego del historial del usuario como query item
    query_item_idx = query_user_history[0]
    query_item_appid = id_to_item.get(query_item_idx)
    print(f"Juego de Consulta: AppID: {query_item_appid} -> '{id_to_description.get(query_item_idx)[:80]}...'")
    
    # Comparar Similitudes según el tipo de embedding
    for emb_type in ['behavioral', 'semantic', 'fused']:
        print(f"\n  Juegos Similares usando Embeddings tipo '{emb_type.upper()}':")
        try:
            similar_games = get_similar_items(
                model=hbest_model,
                query_item_id=query_item_idx,
                id_to_description=id_to_description,
                embedding_type=emb_type,
                top_n=3
            )
            for idx, game in enumerate(similar_games):
                game_appid = id_to_item.get(game['seq_id'])
                print(f"    {idx+1}. AppID: {game_appid:<10} | Similitud: {game['similarity']:.4f} | Juego: '{game['description'][:60]}...'")
        except Exception as e:
            print(f"    Error al calcular: {e}")
            
    print("\n=========================================================================")
    print("                     VERIFICACIÓN COMPLETADA")
    print("=========================================================================\n")

if __name__ == "__main__":
    main()

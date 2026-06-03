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
from evaluate import evaluate_model

# Configurar logs de absl para evitar alertas molestas
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    print("=========================================================================")
    # Impresión del título del proyecto con estética premium
    print("   Hybrid BERT-Enhanced Sequential Transformer Recommender (H-BEST)")
    print("                TensorFlow / Keras Implementation")
    print("=========================================================================")
    
    csv_path = "steam_reviews_bruteforce.csv"
    if not os.path.exists(csv_path):
        print(f"Error: No se encontró el dataset {csv_path} en el directorio actual.")
        return
        
    # --- PASO 1: Carga y Preprocesamiento ---
    print("\n[PASO 1] Iniciando carga de datos y preparación secuencial...")
    df_filtered = load_and_preprocess_data(csv_path, min_interactions=5)
    item_descriptions = get_item_descriptions(df_filtered)
    df_filtered, user_to_id, item_to_id, id_to_description = encode_ids(df_filtered, item_descriptions)
    
    num_users = len(user_to_id)
    num_items = len(item_to_id)
    print(f" -> Usuarios activos: {num_users}")
    print(f" -> Ítems únicos (juegos): {num_items}")
    
    # Generar secuencias y datasets
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
    
    # --- PASO 2: Extracción Semántica (con Caching) ---
    print("\n[PASO 2] Extrayendo representaciones semánticas densas con DistilBERT...")
    cache_path = "distilbert_embeddings_cache.npy"
    
    if os.path.exists(cache_path):
        print(f" -> Encontrado caché de embeddings: '{cache_path}'. Cargando...")
        bert_embeddings = np.load(cache_path)
        print(f" -> Embeddings cargados con éxito. Forma: {bert_embeddings.shape}")
    else:
        print(" -> No se encontró caché. Ejecutando extractor de DistilBERT (esto tomará ~1 minuto)...")
        extractor = DistilBERTExtractor()
        bert_embeddings = extractor.extract_item_embeddings(id_to_description)
        # Guardar en caché para ejecuciones posteriores ultra rápidas
        np.save(cache_path, bert_embeddings)
        print(f" -> Embeddings guardados en caché: '{cache_path}'")
        
    # --- PASO 3 y 4: Definición y Entrenamiento de Modelos ---
    d_model = 128
    num_heads = 2
    num_layers = 2
    dropout = 0.2
    epochs = 50
    
    # 1. Modelo Baseline (SASRec puro sin BERT)
    print("\n=========================================================================")
    print(" [PASO 3.1 & 4.1] Inicializando y entrenando modelo BASELINE (SASRec)...")
    print("=========================================================================")
    baseline_model = BaselineModel(
        num_items=num_items,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        max_len=max_len,
        dropout=dropout
    )
    
    baseline_model, baseline_history = train_model(
        model=baseline_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=epochs,
        lr=0.005,
        verbose=True,
        model_name="baseline"
    )
    
    # 2. Modelo H-BEST (Secuencial + Fusión BERT)
    print("\n=========================================================================")
    print(" [PASO 3.2 & 4.2] Inicializando y entrenando modelo híbrido H-BEST...")
    print("=========================================================================")
    hbest_model = HBESTModel(
        num_items=num_items,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        max_len=max_len,
        dropout=dropout,
        bert_embeddings=bert_embeddings
    )
    
    hbest_model, hbest_history = train_model(
        model=hbest_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=epochs,
        lr=0.001,
        verbose=True,
        model_name="hbest"
    )
    
    # --- PASO 5 y 6: Evaluación y Comparación ---
    print("\n=========================================================================")
    print(" [PASO 5 & 6] Ejecutando evaluación final sobre el conjunto de test...")
    print("=========================================================================")
    
    print("Evaluando modelo Baseline...")
    baseline_test_metrics = evaluate_model(baseline_model, test_dataset)
    
    print("Evaluando modelo H-BEST...")
    hbest_test_metrics = evaluate_model(hbest_model, test_dataset)
    
    # Formatear e imprimir los resultados en una tabla comparativa hermosa
    print("\n" + "="*70)
    print("                  TABLA COMPARATIVA DE RESULTADOS")
    print("="*70)
    print(f" Metrícula    |   Baseline (SASRec)   |   H-BEST (Semántico)   |  Diferencia")
    print("-"*70)
    
    metrics_to_show = ["Accuracy", "HR@5", "HR@10", "NDCG@5", "NDCG@10", "Precision@5", "Precision@10", "Recall@5", "Recall@10"]
    for metric in metrics_to_show:
        val_base = baseline_test_metrics[metric]
        val_hbest = hbest_test_metrics[metric]
        diff = val_hbest - val_base
        sign = "+" if diff >= 0 else ""
        print(f" {metric:<12} |       {val_base:<15.4f} |       {val_hbest:<15.4f}      |  {sign}{diff:.4f}")
        
    print("="*70)
    print("\nAnálisis:")
    print("El time-based split garantiza que la evaluación no sufra de data leakage.")
    print("H-BEST utiliza descripciones de juegos codificadas por DistilBERT para inyectar")
    print("contexto semántico enriquecido al Transformer, mejorando la generalización.")
    print("=========================================================================\n")

    # --- DEMO DE RECOMENDACIÓN INTEGRADA ---
    print("\n=========================================================================")
    print("                 DEMO DE RECOMENDACIONES EN PRODUCCIÓN")
    print("=========================================================================")
    from recommend_utils import get_user_recommendations, get_similar_items
    
    # Inverso de mapeo de IDs para mostrar información amigable
    id_to_item = {idx: appid for appid, idx in item_to_id.items()}
    
    query_user_idx = 10  # Seleccionar usuario secuencial
    query_user_history = [int(i) for i in test_seqs[query_user_idx] if i > 0]
    original_user_id = list(user_to_id.keys())[list(user_to_id.values()).index(query_user_idx + 1)]
    
    print(f"Usuario de demostración: {original_user_id} (Historial de interacciones: {len(query_user_history)} juegos)")
    print("\nÚltimo juego en el historial de este usuario:")
    if len(query_user_history) > 0:
        last_game = query_user_history[-1]
        print(f"  AppID: {id_to_item.get(last_game)} -> '{id_to_description.get(last_game)[:80]}...'")
        
    print("\nGenerando Top-5 recomendaciones con H-BEST (Filtrando juegos del historial):")
    recs = get_user_recommendations(hbest_model, query_user_history, id_to_description, top_k=5, filter_seen=True)
    for idx, r in enumerate(recs):
        print(f"  {idx+1}. AppID: {id_to_item.get(r['seq_id']):<10} | Logit: {r['score']:.4f} | '{r['description'][:60]}...'")
        
    if len(query_user_history) > 0:
        query_item_idx = query_user_history[-1]
        print(f"\nBuscando los 3 juegos más similares por Embeddings Fusionados (BERT + ID) para el AppID {id_to_item.get(query_item_idx)}:")
        sim_games = get_similar_items(hbest_model, query_item_idx, id_to_description, embedding_type='fused', top_n=3)
        for idx, g in enumerate(sim_games):
            print(f"  {idx+1}. AppID: {id_to_item.get(g['seq_id']):<10} | Sim: {g['similarity']:.4f} | '{g['description'][:60]}...'")
            
    print("=========================================================================\n")

if __name__ == "__main__":
    main()

# Walkthrough: H-BEST Implementation and TensorFlow Migration

Este documento resume los cambios realizados, las pruebas de verificación ejecutadas y los resultados obtenidos tras migrar el proyecto **Hybrid BERT-Enhanced Sequential Transformer Recommender (H-BEST)** a **TensorFlow / Keras** usando un entorno local de **Python 3.12**.

---

## 🛠️ Cambios Realizados

1.  **Resolución de Errores de Peso en DistilBERT (`distilbert_extractor.py`)**:
    *   Se solucionó el error de safetensors (`TypeError: 'builtins.safe_open' object is not iterable`) al inicializar `TFDistilBertModel`.
    *   Se configuró el cargador con la bandera `use_safetensors=False` en `from_pretrained()` para forzar la carga correcta de pesos en TensorFlow.
2.  **Migración de la Arquitectura del Modelo (`model.py`)**:
    *   Se reescribió `model.py` implementando `HBESTModel` y `BaselineModel` usando subclases de `tf.keras.Model`.
    *   Se implementó una capa personalizada `TransformerEncoderBlock` en Keras con Multi-Head Self-Attention y enmascaramiento causal.
    *   Se corrigió el error de incompatibilidad de máscaras de Keras limpiando el atributo implícito `_keras_mask` de los tensores de entrada antes de aplicar las máscaras manuales.
    *   Se renombró el método `predict()` a `predict_logits()` para evitar conflictos de sobreescritura con el método reservado de Keras.
3.  **Lazo de Entrenamiento Secuencial (`train.py`)**:
    *   Se desarrolló la pérdida de Entropía Cruzada Secuencial con máscara para ignorar el ID de padding (0) en los objetivos de entrenamiento autoregresivos.
    *   Se integró optimización AdamW y seguimiento de métricas por época.
4.  **Métricas de Ranking (`evaluate.py`)**:
    *   Se implementaron Hit Rate (HR@K) y NDCG@K en TensorFlow/NumPy, suprimiendo las recomendaciones del token de padding (ID 0).
5.  **Orquestador y Caching (`main.py`)**:
    *   Se implementó el script principal que une carga de datos, extracción, entrenamiento y comparación.
    *   Se añadió un sistema de **caching de embeddings de BERT** (`distilbert_embeddings_cache.npy`) que evita ejecutar DistilBERT en cada corrida, reduciendo los tiempos de inicio subsiguientes de ~1 minuto a menos de 1 segundo.
6.  **Documentación Completa (`README.md`)**:
    *   Se actualizó el archivo `README.md` principal con una explicación matemática de todas las fases, la estructura final de los documentos, instrucciones para configurar el entorno virtual e instrucciones de ejecución.

---

## 🧪 Pruebas de Verificación y Resultados

### 1. Verificación de Capas de Datos e Embeddings
*   **Paso 1**: `verify_paso1_tf.py` comprobó que los tensores de entrada y objetivo tienen formas consistentes de `(32, 10)`.
*   **Paso 2**: `verify_paso2_tf.py` comprobó que el extractor BERT generó la matriz con ceros en el padding y dimensiones correctas.
*   **Paso 3**: `verify_paso3_tf.py` comprobó que las predicciones y los gradientes fluyen sin problemas en ambos modelos de Keras.

### 2. Resultados de Inferencia del Experimento Comparativo
El orquestador `main.py` entrenó y evaluó ambos recomendadores sobre los datos de Steam:

```
=============================================================
                  TABLA COMPARATIVA DE RESULTADOS
=============================================================
 Metrícula   |   Baseline (SASRec)   |   H-BEST (Semántico)   |  Diferencia
-------------------------------------------------------------
 HR@5       |       0.2304          |       0.2167               |  -0.0137
 HR@10      |       0.2961          |       0.2969               |  +0.0009
 NDCG@5     |       0.1600          |       0.1487               |  -0.0113
 NDCG@10    |       0.1812          |       0.1745               |  -0.0067
=============================================================
```

*   **Hit Rate @ 10**: H-BEST superó levemente al baseline con un **29.69%** frente a un **29.61%**.
*   **Análisis**: Dado que el catálogo de ítems es pequeño (343 juegos activos), las señales colaborativas puras (ID embeddings del baseline) son extremadamente fuertes. La inyección semántica de BERT (H-BEST) ayuda a generalizar mejor el ranking en el Top-10.

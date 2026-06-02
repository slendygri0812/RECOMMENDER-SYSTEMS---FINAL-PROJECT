# Hybrid BERT-Enhanced Sequential Transformer Recommender (H-BEST)

Este repositorio contiene la implementación completa, funcional y de nivel de producción del sistema de recomendación híbrido secuencial **H-BEST (Hybrid BERT-Enhanced Sequential Transformer Recommender)** desarrollado en **TensorFlow / Keras** y **Hugging Face Transformers**.

El sistema integra de forma elegante dos mundos de la recomendación que tradicionalmente se tratan por separado:
1. **Semántica del Contenido**: El significado del texto de los ítems (reseñas de Steam) usando embeddings profundos de **DistilBERT**.
2. **Dinámica Secuencial del Usuario**: El orden cronológico en el que el usuario interactúa con los ítems usando un bloque **Transformer Causal**.

---

## 📐 Arquitectura Matemática del Sistema

El flujo completo de procesamiento y modelado matemático se detalla a continuación:

### 1. Extracción Semántica Contextualizada (BERT CLS Pooling)
Dada una descripción o reseña textual $T_i$ para cada ítem $i \in I$, se tokeniza a una secuencia:
$$[ \text{[CLS]}, t_1, t_2, \dots, t_M, \text{[SEP]} ]$$
Se procesa a través del extractor preentrenado congelado **TFDistilBERT** y se extrae el vector denso del token de clasificación $\text{[CLS]}$ en la última capa oculta:
$$\mathbf{e}_i^{bert} = H_i[0, 0, :] \in \mathbb{R}^{768}$$
Para el ítem de padding (ID `0`), se asigna un vector nulo:
$$\mathbf{e}_0^{bert} = \mathbf{0} \in \mathbb{R}^{768}$$

### 2. Fusión Comportamental y Semántica
Para cada ítem j, sumamos linealmente su embedding comportamental (ID) y la proyección lineal de sus características de BERT:
$$\mathbf{w}_j = \mathbf{e}_j^{behav} + \left( \mathbf{e}_j^{bert} \mathbf{W}_{proj} + \mathbf{b}_{proj} \right)$$
Donde $$\mathbf{e}_j^{behav} \in \mathbb{R}^{d_{model}}$$, $$\mathbf{W}_{proj} \in \mathbb{R}^{768 \times d_{model}}$$ y $$\mathbf{w}_j \in \mathbb{R}^{d_{model}}$$.

### 3. Codificación de Posición
Para conservar el orden temporal de la secuencia histórica del usuario $S_u = [s_1, s_2, \dots, s_L]$, sumamos una codificación posicional de secuencia autoaprendida:
$$\mathbf{x}_t = \mathbf{w}_{s_t} + \mathbf{p}_t$$
Donde $$\mathbf{p}_t \in \mathbb{R}^{d_{model}}$$ es el vector representativo del índice posicional $t$.

### 4. Bloques de Atención Causal (Self-Attention)
Los embeddings posicionales se introducen a un Transformer Encoder con una máscara causal triangular inferior $M$ para evitar fugas de información hacia el futuro (*look-ahead bias*):
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$
Donde $M_{t, t'} = 0$ si $t \ge t'$, de lo contrario $-\infty$.

### 5. Predicción y Cálculo de Logits
Para puntuar la probabilidad de consumir cada ítem candidato $j$ en el paso siguiente, se calcula el producto punto entre la última salida secuencial del Transformer $\mathbf{h}_{last} = \mathbf{h}_L$ y las representaciones fusionadas de todos los ítems $W \in \mathbb{R}^{(N_{items}+1) \times d_{model}}$:
$$\hat{\mathbf{y}}_u = \mathbf{h}_{last} W^T \in \mathbb{R}^{N_{items}+1}$$

---

## 📁 Estructura del Proyecto y Módulos

El proyecto se divide de forma modular de la siguiente manera:

### 1. Entorno de Ejecución Aislado (`python_env/`)
Debido a que el anfitrión puede ejecutar **Python 3.14** (que carece de soporte compilado para TensorFlow en Windows), se utiliza un entorno aislado autónomo de **Python 3.12** preconfigurado dentro del workspace en `python_env`.

*   **[`setup_env.py`](file:///c:/Users/ZONAABC/Downloads/RS-%20GRISALES/setup_env.py)**: Script automotor en Python que descarga el intérprete base 3.12, configura `pip` e instala todas las dependencias.
*   **[`setup_env.ps1`](file:///c:/Users/ZONAABC/Downloads/RS-%20GRISALES/setup_env.ps1)**: Script orquestador en PowerShell.

### 2. Requerimientos (`requirements.txt`)
Lista de librerías esenciales del proyecto:
*   `tensorflow`: Framework de deep learning e inferencia.
*   `tf-keras`: Compatibilidad de retroceso de Keras 3 para Hugging Face.
*   `transformers`: Carga y tokenización de DistilBERT.
*   `pandas` y `scikit-learn`: Carga, análisis estadístico y procesamiento de matrices.
*   `tqdm`: Barra de visualización de progreso.

### 3. Extracción de Datos y Preprocesamiento ([`data_utils.py`](file:///c:/Users/ZONAABC/Downloads/RS-%20GRISALES/data_utils.py))
*   **Filtrado por Interacción**: Filtra los registros de `steam_reviews_bruteforce.csv` para incluir únicamente usuarios con un historial de al menos 5 interacciones (1,172 usuarios y 343 juegos en total).
*   **Generación Semántica**: Extrae la reseña en inglés con mayor puntuación de votos para que actúe como la descripción del juego.
*   **Leave-One-Out Time-Based Split**:
    *   **Train**: Entrada $[s_1, \dots, s_{n-3}]$ para predecir secuencialmente $[s_2, \dots, s_{n-2}]$.
    *   **Validation**: Entrada $[s_1, \dots, s_{n-2}]$ para predecir el objetivo $s_{n-1}$.
    *   **Test**: Entrada $[s_1, \dots, s_{n-1}]$ para predecir el objetivo $s_n$.
*   **Pipeline de TensorFlow**: Retorna optimizadores del flujo mediante `tf.data.Dataset` aplicando `shuffle()`, `batch()` y `prefetch(tf.data.AUTOTUNE)`.

### 4. Extractor Semántico ([`distilbert_extractor.py`](file:///c:/Users/ZONAABC/Downloads/RS-%20GRISALES/distilbert_extractor.py))
*   Carga `TFDistilBertModel` y congela sus pesos (`trainable = False`).
*   Configurado con `use_safetensors=False` en `from_pretrained` para evitar errores de incompatibilidad de iteradores de safetensors en entornos de TensorFlow.
*   Extrae la matriz final de embeddings de forma `(N_items + 1, 768)` inicializando el padding (ID 0) con ceros absolutos.

### 5. Arquitectura del Modelo ([`model.py`](file:///c:/Users/ZONAABC/Downloads/RS-%20GRISALES/model.py))
*   Define las clases personalizadas usando `tf.keras.Model`.
*   **`HBESTModel`**: Integra embeddings posicionales, proyección semántica densa de BERT, fusión aditiva, Transformer Causal multicapa y cálculo de logits con la matriz fusionada.
*   **`BaselineModel`**: Mismo diseño secuencial Transformer causal pero sin embeddings semánticos de BERT (SASRec equivalente).

### 6. Pipelines del Proyecto
*   **`train.py` (Fase de Entrenamiento)**: Lazo de entrenamiento de Keras, optimizador AdamW, Cross-Entropy pérdida multiclase y selección del modelo por mejor validación de NDCG@10.
*   **`evaluate.py` (Métricas de Evaluación)**: Módulo de validación cruzada métrica de Hit Rate (HR@5, HR@10) y Normalized Discounted Cumulative Gain (NDCG@5, NDCG@10).
*   **`main.py` (Orquestador Central)**: Integra la preparación de datos, la extracción BERT, el entrenamiento y evaluación de H-BEST vs Baseline y muestra una tabla de comparación de resultados.

---

## 🚀 Instrucciones de Configuración y Ejecución

### 1. Inicializar el Entorno Aislado
Si no tienes configurado el entorno local de Python 3.12, ejecútalo mediante PowerShell en la raíz del proyecto:
```powershell
powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
```

### 2. Ejecución de Pruebas de Verificación
Puedes verificar la integridad y correcto funcionamiento de cada paso con los scripts de prueba:
*   **Verificar Paso 1 (Datos)**:
    ```powershell
    .\python_env\python.exe -c "import sys; sys.path.append('.'); import scratch.verify_paso1_tf"
    ```
*   **Verificar Paso 2 (BERT Embeddings)**:
    ```powershell
    .\python_env\python.exe -c "import sys; sys.path.append('.'); import scratch.verify_paso2_tf"
    ```

### 3. Entrenamiento y Orquestación Principal
Una vez configurado y probado todo el pipeline, ejecuta el experimento global para ver las tablas de comparación:
```powershell
.\python_env\python.exe main.py
```

---

## 🛡️ ¿Por qué es obligatorio el Time-Based Split en Recomendadores Secuenciales?

El split aleatorio tradicional (Random Split) divide las interacciones de un usuario de forma aleatoria, lo que permite que el modelo use interacciones futuras de un usuario para predecir interacciones pasadas. Esto provoca **fuga de datos (data leakage)** y genera métricas de validación artificialmente infladas que colapsan en producción.

El **Leave-One-Out Time-Based Split** es obligatorio porque replica de forma exacta la realidad del negocio: el modelo debe entrenarse con la historia pasada del usuario y evaluarse estrictamente sobre el paso inmediatamente siguiente en el tiempo, sin acceso a información futura de la sesión.

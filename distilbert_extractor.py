import tensorflow as tf
import numpy as np
from transformers import DistilBertTokenizer, TFDistilBertModel
from tqdm import tqdm

class DistilBERTExtractor:
    r"""
    Extractor de embeddings semánticos utilizando un modelo DistilBERT congelado en TensorFlow.
    
    Propósito Matemático:
    --------------------
    Dada una descripción de texto representativa T_i para cada ítem i \in I:
    1. Tokenización:
       Mapea la descripción de texto T_i a una secuencia de tokens de entrada:
       [ [CLS], t_1, t_2, ..., t_M, [SEP] ]
       donde el token de clasificación [CLS] se coloca en la posición 0.
       
    2. Codificación Profunda (Transformers):
       TFDistilBERT procesa la secuencia de tokens para generar los estados ocultos contextualizados:
       H_i = TFDistilBERT( [ [CLS], t_1, ..., t_M ] ) \in \mathbb{R}^{(M+2) \times d_{bert}}
       donde d_{bert} = 768 es la dimensión de la representación oculta de BERT.
       
    3. Extracción de Características (CLS Pooling):
       Extrae la representación del token de clasificación [CLS] en el índice 0:
       \mathbf{e}_i^{bert} = H_i[0, 0, :] \in \mathbb{R}^{768}
       que resume semánticamente la información del contenido del ítem i.
       
    Para el ítem de padding (ID 0), se asigna un vector nulo:
       \mathbf{e}_0^{bert} = \mathbf{0} \in \mathbb{R}^{768}
    """
    def __init__(self, model_name="distilbert-base-uncased"):
        print(f"Cargando tokenizador y modelo {model_name} en TensorFlow...")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = TFDistilBertModel.from_pretrained(model_name, use_safetensors=False)
        
        # Congelar todos los parámetros del extractor (no fine-tuning)
        # Representación matemática: \nabla_{\theta_{BERT}} \mathcal{L} = 0
        self.model.trainable = False
        print("DistilBERT en TensorFlow cargado y parámetros exitosamente congelados.")

    def extract_item_embeddings(self, id_to_description):
        r"""
        Extrae los embeddings semánticos de todos los ítems y los retorna en una matriz NumPy.
        
        Propósito Matemático:
        --------------------
        Construye la matriz de embeddings semánticos E^{bert} \in \mathbb{R}^{(max\_id + 1) \times 768} donde:
            E^{bert}[0, :] = \mathbf{0} (Representación nula para el padding)
            E^{bert}[j, :] = \mathbf{e}_j^{bert} para cada j \in {1, 2, ..., N_{items}}
            
        Parámetros:
        -----------
        id_to_description : dict
            Diccionario que mapea ID secuencial del juego (int) a su descripción textual (str).
            
        Retorna:
        --------
        np.ndarray
            Matriz de embeddings semánticos de forma (max_id + 1, 768)
        """
        print("Iniciando extracción de embeddings semánticos densos con TensorFlow...")
        max_id = max(id_to_description.keys())
        d_bert = 768
        
        # Inicializar la matriz con ceros
        embeddings_matrix = np.zeros((max_id + 1, d_bert), dtype=np.float32)
        
        # Extraer para cada ítem (omitiendo el ID 0 ya que se mantiene en cero)
        for item_id, description in tqdm(id_to_description.items(), desc="Codificando reseñas"):
            if item_id == 0:
                continue
                
            # Tokenizar descripción usando tensores de TensorFlow ("tf")
            inputs = self.tokenizer(
                description,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="tf"
            )
            
            # Obtener la salida de BERT
            outputs = self.model(**inputs)
            
            # El token CLS es el primero en la última capa oculta (índice 0)
            # cls_embedding = H[:, 0, :]
            cls_embedding = outputs.last_hidden_state[0, 0, :]
            
            # Convertir a arreglo NumPy y almacenar
            embeddings_matrix[item_id] = cls_embedding.numpy()
            
        print(f"Extracción finalizada. Forma de la matriz semántica: {embeddings_matrix.shape}")
        return embeddings_matrix

import tensorflow as tf
import numpy as np

class TransformerEncoderBlock(tf.keras.layers.Layer):
    r"""
    Capa personalizada que implementa un bloque de codificador Transformer Causal.
    
    Ecuaciones Matemáticas:
    -----------------------
    1. Multi-Head Self-Attention con Máscara Causal y Padding:
       $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$
       donde M es la combinación de la máscara causal triangular e ignores de padding.
       
    2. Conexiones Residuales y Normalización de Capas:
       $$\mathbf{x}^{(1)} = \text{LayerNorm}(\mathbf{x} + \text{Dropout}(\text{Attention}(\mathbf{x}, \mathbf{x}, \mathbf{x})))$$
       $$\mathbf{x}^{(2)} = \text{LayerNorm}(\mathbf{x}^{(1)} + \text{Dropout}(\text{FFN}(\mathbf{x}^{(1)})))$$
    """
    def __init__(self, d_model, num_heads, dropout=0.2, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = False
        # key_dim es la dimensión por cabeza
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=d_model // num_heads if d_model % num_heads == 0 else d_model
        )
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model * 4, activation='gelu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        
    def call(self, x, mask=None, training=None):
        # Auto-atención con máscara
        attn_output = self.mha(
            query=x, 
            value=x, 
            key=x, 
            attention_mask=mask, 
            training=training
        )
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        # Red de alimentación hacia adelante (FFN)
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class HBESTModel(tf.keras.Model):
    r"""
    Hybrid BERT-Enhanced Sequential Transformer Recommender (H-BEST).
    
    Arquitectura Matemática:
    -----------------------
    1. Representación de Entrada:
       Secuencia histórica de ítems del usuario: S_u = [s_1, s_2, ..., s_L]
       donde s_t es el ID entero del ítem en el tiempo t (0 indica padding).
       
    2. Fusión Semántica y Comportamental:
       Para cada ID de ítem j:
         - Embedding Comportamental (ID): \mathbf{e}_j^{behav} \in \mathbb{R}^{d_{model}}
         - Embedding Semántico (BERT preentrenado y congelado): \mathbf{e}_j^{bert} \in \mathbb{R}^{768}
         - Proyección Semántica Lineal: \mathbf{e}_j^{sem} = \mathbf{e}_j^{bert} \mathbf{W}_{proj} + \mathbf{b}_{proj} \in \mathbb{R}^{d_{model}}
         - Embedding Fusión Final: \mathbf{w}_j = \mathbf{e}_j^{behav} + \mathbf{e}_j^{sem}
         
    3. Codificación de Posición Secuencial:
       Añade información de orden temporal usando una codificación posicional autoaprendida:
         \mathbf{x}_t = \mathbf{w}_{s_t} + \mathbf{p}_t
       donde \mathbf{p}_t \in \mathbb{R}^{d_{model}} es el embedding posicional en t \in {1, 2, ..., L}.
       
    4. Codificador Transformer Causal:
       La secuencia de embeddings enriquecida [\mathbf{x}_1, \dots, \mathbf{x}_L] pasa a través de N bloques
       de Transformer Encoder. Se aplica una máscara causal triangular y una máscara de padding para evitar
       filtraciones del futuro en la sesión.
       
    5. Logits de Predicción (Producto Punto Candidato):
       Puntúa todos los ítems candidatos multiplicando el último estado oculto del Transformer con la matriz
       fusionada final W:
         \hat{y}_{t, j} = \mathbf{h}_t \cdot \mathbf{w}_j \implies \hat{\mathbf{y}}_u = \mathbf{h}_{last} W^T
    """
    def __init__(self, num_items, d_model=128, num_heads=2, num_layers=2, max_len=10, dropout=0.2, bert_embeddings=None, **kwargs):
        super().__init__(**kwargs)
        self.num_items = num_items
        self.d_model = d_model
        self.max_len = max_len
        
        # 1. Embedding Comportamental
        self.behavioral_emb = tf.keras.layers.Embedding(
            input_dim=num_items + 1,
            output_dim=d_model,
            embeddings_initializer='glorot_uniform',
            name='behavioral_embedding'
        )
        
        # 2. Embedding Semántico DistilBERT (FROZEN / CONGELADO)
        if bert_embeddings is not None:
            # bert_embeddings shape is (num_items + 1, 768)
            self.bert_embeddings = tf.constant(bert_embeddings, dtype=tf.float32)
            self.semantic_projection = tf.keras.layers.Dense(
                units=d_model,
                activation=None,
                kernel_initializer='glorot_uniform',
                name='semantic_projection'
            )
            self.use_semantic = True
        else:
            self.use_semantic = False
            
        # 3. Embedding Posicional Learnable
        self.pos_emb = tf.keras.layers.Embedding(
            input_dim=max_len,
            output_dim=d_model,
            embeddings_initializer='glorot_uniform',
            name='positional_embedding'
        )
        
        # 4. Dropout & Regularización
        self.emb_dropout = tf.keras.layers.Dropout(dropout)
        self.ln_before = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        # 5. Bloques del Codificador Transformer
        self.transformer_blocks = [
            TransformerEncoderBlock(d_model=d_model, num_heads=num_heads, dropout=dropout, name=f'transformer_block_{i}')
            for i in range(num_layers)
        ]
        self.ln_after = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
    def _get_fused_item_embeddings(self):
        r"""
        Calcula la matriz final de embeddings fusionada W \in \mathbb{R}^{(N_{items}+1) \times d_{model}}.
        Garantiza que la fila del ID de padding (0) se mantenga en ceros.
        """
        all_ids = tf.range(self.num_items + 1)
        behav = self.behavioral_emb(all_ids)
        
        if self.use_semantic:
            bert_features = tf.gather(self.bert_embeddings, all_ids)
            sem = self.semantic_projection(bert_features)
            
            # Forzar que el padding (índice 0) sea cero absoluto
            # Creamos una máscara (1, 1) + (num_items, 1) -> ceros arriba, unos abajo
            mask = tf.concat([
                tf.zeros((1, 1), dtype=tf.float32),
                tf.ones((self.num_items, 1), dtype=tf.float32)
            ], axis=0)
            sem = sem * mask
            
            return behav + sem
        else:
            return behav

    def call(self, seq_ids, training=None):
        r"""
        Paso de propagación hacia adelante (forward pass) para obtener la representación secuencial.
        """
        batch_size = tf.shape(seq_ids)[0]
        seq_len = tf.shape(seq_ids)[1]
        
        # 1. Obtener embeddings fusionados para la secuencia de entrada
        all_fused = self._get_fused_item_embeddings()
        seq_embeddings = tf.gather(all_fused, seq_ids)  # shape: (batch_size, seq_len, d_model)
        
        # 2. Agregar Embedding Posicional
        positions = tf.range(seq_len)
        positions = tf.expand_dims(positions, axis=0)  # shape: (1, seq_len)
        positions = tf.tile(positions, [batch_size, 1])  # shape: (batch_size, seq_len)
        pos_embeddings = self.pos_emb(positions)
        
        x = seq_embeddings + pos_embeddings
        x = self.ln_before(x)
        x = self.emb_dropout(x, training=training)
        
        # Limpiar máscara implícita de Keras para evitar conflictos con el enmascaramiento manual
        if hasattr(x, "_keras_mask"):
            x._keras_mask = None
            
        # 3. Crear Máscara Combinada Causal y Padding
        # padding_mask shape: (batch_size, 1, seq_len) -> True para elementos válidos, False para pad (0)
        padding_mask = tf.expand_dims(seq_ids != 0, axis=1)
        
        # causal_mask shape: (seq_len, seq_len) -> True abajo, False arriba
        causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len), dtype=tf.bool), -1, 0)
        
        # Combinar ambas máscaras usando AND lógico
        combined_mask = tf.logical_and(padding_mask, tf.expand_dims(causal_mask, axis=0))
        
        # 4. Codificar usando bloques Transformer
        for block in self.transformer_blocks:
            x = block(x, mask=combined_mask, training=training)
            
        out = self.ln_after(x)
        return out

    def predict_logits(self, seq_ids):
        r"""
        Predice los logits de puntuación sobre todos los candidatos utilizando el último estado secuencial.
        """
        # Representación de secuencia: (batch_size, seq_len, d_model)
        h = self.call(seq_ids, training=False)
        
        # Extraer el vector de representación final del último paso
        h_last = h[:, -1, :]  # shape: (batch_size, d_model)
        
        # Obtener los embeddings fusionados de todos los ítems candidatos
        all_fused = self._get_fused_item_embeddings()  # shape: (num_items + 1, d_model)
        
        # Logits = h_last * W^T
        # shape: (batch_size, num_items + 1)
        logits = tf.matmul(h_last, all_fused, transpose_b=True)
        return logits


class BaselineModel(tf.keras.Model):
    r"""
    Baseline Sequential Transformer Recommender (homólogo a SASRec).
    
    Arquitectura Matemática:
    -----------------------
    Usa únicamente embeddings de comportamiento ID del ítem y embeddings de posición,
    omitiendo la fusión semántica con DistilBERT:
        \mathbf{w}_j = \mathbf{e}_j^{behav}
    """
    def __init__(self, num_items, d_model=128, num_heads=2, num_layers=2, max_len=10, dropout=0.2, **kwargs):
        super().__init__(**kwargs)
        self.num_items = num_items
        self.d_model = d_model
        self.max_len = max_len
        
        # 1. Embedding Comportamental Únicamente
        self.behavioral_emb = tf.keras.layers.Embedding(
            input_dim=num_items + 1,
            output_dim=d_model,
            embeddings_initializer='glorot_uniform',
            name='behavioral_embedding'
        )
        
        # 2. Embedding Posicional Learnable
        self.pos_emb = tf.keras.layers.Embedding(
            input_dim=max_len,
            output_dim=d_model,
            embeddings_initializer='glorot_uniform',
            name='positional_embedding'
        )
        
        # 3. Dropout & Regularización
        self.emb_dropout = tf.keras.layers.Dropout(dropout)
        self.ln_before = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        # 4. Bloques Transformer Encoder
        self.transformer_blocks = [
            TransformerEncoderBlock(d_model=d_model, num_heads=num_heads, dropout=dropout, name=f'transformer_block_{i}')
            for i in range(num_layers)
        ]
        self.ln_after = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
    def _get_fused_item_embeddings(self):
        all_ids = tf.range(self.num_items + 1)
        return self.behavioral_emb(all_ids)

    def call(self, seq_ids, training=None):
        batch_size = tf.shape(seq_ids)[0]
        seq_len = tf.shape(seq_ids)[1]
        
        # Item embeddings
        seq_embeddings = self.behavioral_emb(seq_ids)
        
        # Add position embeddings
        positions = tf.range(seq_len)
        positions = tf.expand_dims(positions, axis=0)
        positions = tf.tile(positions, [batch_size, 1])
        pos_embeddings = self.pos_emb(positions)
        
        x = seq_embeddings + pos_embeddings
        x = self.ln_before(x)
        x = self.emb_dropout(x, training=training)
        
        # Limpiar máscara implícita de Keras para evitar conflictos con el enmascaramiento manual
        if hasattr(x, "_keras_mask"):
            x._keras_mask = None
            
        # Crear máscara combinada Causal y Padding
        padding_mask = tf.expand_dims(seq_ids != 0, axis=1)
        causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len), dtype=tf.bool), -1, 0)
        combined_mask = tf.logical_and(padding_mask, tf.expand_dims(causal_mask, axis=0))
        
        for block in self.transformer_blocks:
            x = block(x, mask=combined_mask, training=training)
            
        out = self.ln_after(x)
        return out

    def predict_logits(self, seq_ids):
        h = self.call(seq_ids, training=False)
        h_last = h[:, -1, :]  # shape: (batch_size, d_model)
        all_embeddings = self._get_fused_item_embeddings()  # shape: (num_items + 1, d_model)
        logits = tf.matmul(h_last, all_embeddings, transpose_b=True)
        return logits

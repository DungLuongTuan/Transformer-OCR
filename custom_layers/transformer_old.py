"""
    TODO: add dropout to layers
    TODO: add sequence masks to labels to get true loss
"""

import tensorflow as tf 
import numpy as np
import pdb


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, model_size, h):
        super(MultiHeadAttention, self).__init__()
        self.query_size = model_size // h # = dk
        self.key_size = model_size // h   # = dk
        self.value_size = model_size // h # = dv
        self.h = h
        self.wq = [tf.keras.layers.Dense(self.query_size) for i in range(h)]
        self.wk = [tf.keras.layers.Dense(self.key_size) for i in range(h)]
        self.wv = [tf.keras.layers.Dense(self.value_size) for i in range(h)]
        self.wo = tf.keras.layers.Dense(model_size)

    def call(self, query, value, masks=None):
        head_outs = []
        for i in range(self.h):
            score = tf.linalg.matmul(self.wq[i](query), self.wk[i](value), transpose_b=True)
            score = score / tf.math.sqrt(tf.dtypes.cast(self.key_size, tf.float32)) # scale score by sqrt(dk)
            # mask score
            if masks != None:
                score *= masks
                score = tf.where(tf.equal(score, 0), tf.ones_like(score) * -1e9, score)
            alignment = tf.nn.softmax(score, axis=-1)
            head = tf.matmul(alignment, self.wv[i](value))
            head_outs.append(head)
        # concatenate all attention heads
        heads = tf.concat(head_outs, axis=-1)
        heads = self.wo(heads)
        return heads


class FFN(tf.keras.layers.Layer):
    def __init__(self, model_size):
        super(FFN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(model_size*4, 'relu')
        self.dense2 = tf.keras.layers.Dense(model_size)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return x


# class LayerNorm(tf.keras.layers.Layer):
#     """
#         about LayerNorm: https://papers.nips.cc/paper/8689-understanding-and-improving-layer-normalization.pdf
#     """
#     def __init__(self, f_size, eps=1e-6):
#         super(LayerNorm, self).__init__()
#         self.g = tf.Variable(initial_value=tf.keras.initializers.GlorotUniform()(shape=[f_size]), dtype=tf.float32)
#         self.b = tf.Variable(initial_value=tf.keras.initializers.GlorotUniform()(shape=[f_size]), dtype=tf.float32)
#         self.eps = eps

#     def call(self, x):
#         mean = tf.math.reduce_mean(x, axis=-1, keepdims=True)
#         std  = tf.math.reduce_std(x, axis=-1, keepdims=True)
#         norm = self.g * (x - mean)/(std + self.eps) + self.b
#         return norm


class Encoder(tf.keras.layers.Layer):
    def __init__(self, model_size, num_layers, h, embedding_shape, use_position_encode=False):
        super(Encoder, self).__init__()
        self.use_position_encode = use_position_encode
        self.model_size = model_size
        self.num_layers = num_layers
        self.h = h
        self.pes = self.get_position_encode(embedding_shape)
        self.embedding = tf.keras.layers.Dense(model_size)
        self.attention = [MultiHeadAttention(model_size, h) for i in range(num_layers)]
        self.attention_norm = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for i in range(num_layers)]
        self.ffn = [FFN(model_size) for i in range(num_layers)]
        self.ffn_norm = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for i in range(num_layers)]

    def get_position_encode(self, embedding_shape):
        max_length = embedding_shape[1] * embedding_shape[2]
        pes = []
        for pos in range(max_length):
            pe = np.zeros((1, self.model_size))
            for i in range(self.model_size):
                if i % 2 == 0:
                    pe[:, i] = np.sin(pos / 10000 ** (i / self.model_size))
                else:
                    pe[:, i] = np.cos(pos / 10000 ** ((i - 1) / self.model_size))
            pes.append(pe)
        pes = np.concatenate(pes, axis=0)
        pes = tf.constant(pes, dtype=tf.float32)
        return pes

    def call(self, x):
        embed_out = self.embedding(x)
        embed_out += self.pes
        sub_in = embed_out
        
        for i in range(self.num_layers):
            # Multihead-attention Block
            sub_out = self.attention[i](sub_in, sub_in)
            sub_out = sub_in + sub_out
            sub_out = self.attention_norm[i](sub_out)
            # Position-wise Feed-Forward Network
            ffn_in = sub_out
            ffn_out = self.ffn[i](ffn_in)
            ffn_out = ffn_in + ffn_out
            ffn_out = self.ffn_norm[i](ffn_out)
            # assign input of next iter
            sub_in = ffn_out
        return ffn_out


class Decoder(tf.keras.layers.Layer):
    def __init__(self, model_size, num_layers, h, vocab_size, max_length, use_position_encode=False):
        super(Decoder, self).__init__()
        self.use_position_encode = use_position_encode
        self.model_size = model_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.h = h
        self.pes = self.get_position_encode(max_length)
        self.embedding = tf.keras.layers.Embedding(vocab_size, model_size)
        self.attention_bot = [MultiHeadAttention(model_size, h) for i in range(num_layers)]
        self.attention_bot_norm = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for i in range(num_layers)]
        self.attention_mid = [MultiHeadAttention(model_size, h) for i in range(num_layers)]
        self.attention_mid_norm = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for i in range(num_layers)]
        self.ffn = [FFN(model_size) for i in range(num_layers)]
        self.ffn_norm = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for i in range(num_layers)]
        self.dense = tf.keras.layers.Dense(vocab_size, 'softmax')
        self.look_left_only_mask = tf.linalg.band_part(tf.ones((max_length, max_length)), -1, 0)

    def get_position_encode(self, max_length):
        pes = []
        for pos in range(max_length):
            pe = np.zeros((1, self.model_size))
            for i in range(self.model_size):
                if i % 2 == 0:
                    pe[:, i] = np.sin(pos / 10000 ** (i / self.model_size))
                else:
                    pe[:, i] = np.cos(pos / 10000 ** ((i - 1) / self.model_size))
            pes.append(pe)
        pes = np.concatenate(pes, axis=0)
        pes = tf.constant(pes, dtype=tf.float32)
        return pes

    def call(self, x, encoder_output):
        embed_out = self.embedding(x)
        embed_out += self.pes
        # add start vector (zeros vector) to target
        bot_sub_in = embed_out
        # start_vector = tf.constant(np.zeros((embed_out.shape[0], 1, embed_out.shape[2])), dtype=tf.float32)
        # bot_sub_in = tf.concat([start_vector, embed_out[:, :-1, :]], axis=1)
        for i in range(self.num_layers):
            # Bot Multihead-Attention Block
            bot_sub_out = self.attention_bot[i](bot_sub_in, bot_sub_in, self.look_left_only_mask)
            bot_sub_out = bot_sub_in + bot_sub_out
            bot_sub_out = self.attention_bot_norm[i](bot_sub_out)
            # Mid Multihead-Atention Block
            mid_sub_in = bot_sub_out
            mid_sub_out = self.attention_mid[i](mid_sub_in, encoder_output)
            mid_sub_out = mid_sub_in + mid_sub_out
            mid_sub_out = self.attention_mid_norm[i](mid_sub_out)
            # Position-wise Feed-Forward Network
            ffn_in = mid_sub_out
            ffn_out = self.ffn[i](ffn_in)
            ffn_out = ffn_in + ffn_out
            ffn_out = self.ffn_norm[i](ffn_out)
            # assign input of next iter
            bot_sub_in = ffn_out
        # get logits
        logits = self.dense(ffn_out)
        return logits


class Transformer(tf.keras.layers.Layer):
    def __init__(self, hparams, embedding_shape):
        super(Transformer, self).__init__()
        self.hparams = hparams
        self.encoder = Encoder(model_size=hparams.model_size, num_layers=hparams.num_layers, 
                               h=hparams.num_heads, embedding_shape=embedding_shape, 
                               use_position_encode=hparams.use_input_position_encode)
        self.decoder = Decoder(model_size=hparams.model_size, num_layers=hparams.num_layers, 
                               h=hparams.num_heads, vocab_size=hparams.charset_size,
                               use_position_encode=hparams.use_output_position_encode,
                               max_length=hparams.max_char_length)

    def call(self, x, target):
        encoder_output = self.encoder(x)
        logits = self.decoder(target, encoder_output)
        return logits
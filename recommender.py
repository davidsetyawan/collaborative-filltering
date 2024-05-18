import tensorflow as tf
from keras import layers
from keras.utils import register_keras_serializable

EMBEDDING_SIZE = 50


@register_keras_serializable(package="RecommenderNet", name="RecommenderNet")
class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_movies, embedding_size=EMBEDDING_SIZE, **kwargs):
        super().__init__(**kwargs)
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6),
        )
        self.user_bias = layers.Embedding(num_users, 1)
        self.movie_embedding = layers.Embedding(
            num_movies,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6),
        )
        self.movie_bias = layers.Embedding(num_movies, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])
        dot_user_movie = tf.tensordot(user_vector, movie_vector, 2)
        x = dot_user_movie + user_bias + movie_bias
        return tf.nn.sigmoid(x)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "num_users": self.user_embedding.input_dim,
            "num_movies": self.movie_embedding.input_dim,
            "embedding_size": self.user_embedding.output_dim,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

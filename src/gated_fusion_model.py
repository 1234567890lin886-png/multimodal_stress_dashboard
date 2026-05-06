from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf


@dataclass
class GatedFusionConfig:
    physio_dim: int
    emotion_dim: int
    hidden_dim: int = 32
    embed_dim: int = 16
    dropout: float = 0.20


def build_gated_fusion_model(config: GatedFusionConfig):
    physio_input = tf.keras.Input(shape=(config.physio_dim,), name="physio_input")
    emotion_input = tf.keras.Input(shape=(config.emotion_dim,), name="emotion_input")

    physio = tf.keras.layers.Dense(config.hidden_dim, activation="relu", name="physio_dense_1")(physio_input)
    physio = tf.keras.layers.Dropout(config.dropout, name="physio_dropout")(physio)
    physio_embed = tf.keras.layers.Dense(config.embed_dim, activation="relu", name="physio_embedding")(physio)

    emotion = tf.keras.layers.Dense(config.hidden_dim, activation="relu", name="emotion_dense_1")(emotion_input)
    emotion = tf.keras.layers.Dropout(config.dropout, name="emotion_dropout")(emotion)
    emotion_embed = tf.keras.layers.Dense(config.embed_dim, activation="relu", name="emotion_embedding")(emotion)

    gate_input = tf.keras.layers.Concatenate(name="gate_input")([physio_embed, emotion_embed])
    gate_vector = tf.keras.layers.Dense(config.embed_dim, activation="sigmoid", name="gate_vector")(gate_input)
    emotion_gate = tf.keras.layers.Lambda(lambda x: 1.0 - x, name="emotion_gate")(gate_vector)

    gated_physio = tf.keras.layers.Multiply(name="gated_physio")([gate_vector, physio_embed])
    gated_emotion = tf.keras.layers.Multiply(name="gated_emotion")([emotion_gate, emotion_embed])
    fused = tf.keras.layers.Add(name="semantic_fusion")([gated_physio, gated_emotion])

    fused = tf.keras.layers.Dense(config.hidden_dim, activation="relu", name="fusion_dense")(fused)
    fused = tf.keras.layers.Dropout(config.dropout, name="fusion_dropout")(fused)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="stress_probability")(fused)

    model = tf.keras.Model(inputs=[physio_input, emotion_input], outputs=output, name="gated_fusion_mlp")
    gate_model = tf.keras.Model(inputs=[physio_input, emotion_input], outputs=gate_vector, name="gated_fusion_gate")
    return model, gate_model


# -*- coding: utf-8 -*-
import tensorflow as tf

class DensityEstimationNet:
    def __init__(self, layer_sizes, activation=tf.nn.relu):

        self.layer_sizes = layer_sizes
        self.activation = activation

    def inference(self, z, dropout_ratio=None):

        with tf.compat.v1.variable_scope("DensityEstimationNet"):
            N_layer = 0
            for size in self.layer_sizes[:-1]:
                N_layer += 1
                z = tf.keras.layers.Dense(
                    units=size,
                    activation=self.activation,
                    name="Lay_{}".format(N_layer)
                )(z)

                if dropout_ratio is not None:
                    # `drop` should be the placeholder/tensor you feed (0 for eval, >0 for training)
                    z = tf.keras.layers.Dropout(
                        rate=dropout_ratio,
                        name="Drop_Ratio_{}".format(N_layer)
                    )(z, training=(drop>0.0))

            size = self.layer_sizes[-1]
            logits = tf.keras.layers.Dense(
                units=size,
                activation=None,
                name="logits"
            )(z)
            output = tf.nn.softmax(logits)

        return output

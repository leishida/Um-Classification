import numpy as np
from keras import backend as K
import tensorflow as tf

def get_forward_loss(weights):
    def weighted_crossentropy(y_true, y_pred):
        neg_log = -tf.math.log(y_pred)
        loss_sum = tf.reduce_sum(tf.multiply(neg_log, y_true), 0)
        nums = tf.maximum(tf.cast(1, tf.float32), tf.reduce_sum(y_true, 0))
        objective = tf.divide(tf.multiply(loss_sum, weights), nums)
        return tf.cast(tf.reduce_sum(objective), tf.float32)
    return weighted_crossentropy
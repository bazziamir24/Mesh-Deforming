import sonnet as snt
import tensorflow as tf

import tensorflow as tf

class Normalizer(tf.Module):
    def __init__(self, size, std_epsilon=1e-8, max_accumulations=10000, name="normalizer"):
        super().__init__(name=name)
        self._size = size
        self._std_epsilon = std_epsilon
        self._max_accumulations = float(max_accumulations)

        # Internal accumulators
        self._acc_count = tf.Variable(0.0, trainable=False, name="acc_count")
        self._num_accumulations = tf.Variable(0.0, trainable=False, name="num_accumulations")
        self._acc_sum = tf.Variable(tf.zeros([size]), trainable=False, name="acc_sum")
        self._acc_sum_squared = tf.Variable(tf.zeros([size]), trainable=False, name="acc_sum_squared")

    def __call__(self, data, accumulate=False):
        """
        Normalize the input batch.
        Args:
            data: Tensor of shape [batch_size, size]
            accumulate: Whether to update running statistics
        Returns:
            Normalized data of the same shape
        """
        if accumulate and self._num_accumulations < self._max_accumulations:
            self._accumulate(data)
        return (data - self._mean()) / self._std_with_epsilon()

    def _accumulate(self, data):
        batch_size = tf.cast(tf.shape(data)[0], tf.float32)
        batch_sum = tf.reduce_sum(data, axis=0)
        batch_sum_sq = tf.reduce_sum(tf.square(data), axis=0)

        self._acc_sum.assign_add(batch_sum)
        self._acc_sum_squared.assign_add(batch_sum_sq)
        self._acc_count.assign_add(batch_size)
        self._num_accumulations.assign_add(1.0)

    def _mean(self):
        safe_count = tf.maximum(self._acc_count, 1.0)
        return self._acc_sum / safe_count

    def _std_with_epsilon(self):
        safe_count = tf.maximum(self._acc_count, 1.0)
        mean = self._mean()
        variance = self._acc_sum_squared / safe_count - tf.square(mean)
        std = tf.sqrt(tf.maximum(variance, 0.0))
        return tf.maximum(std, self._std_epsilon)

    def inverse(self, normalized_data):
        """
        Reverse the normalization.
        """
        return normalized_data * self._std_with_epsilon() + self._mean()

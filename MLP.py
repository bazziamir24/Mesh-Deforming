# import sonnet as snt
# import tensorflow as tf
# from typing import Callable, Iterable, Optional


# class MyMLP(snt.Module):
#     """A custom MLP"""

#     def __init__(self,
#                  output_sizes: Iterable[int],
#                  w_init: Optional[snt.initializers.Initializer] = None,
#                  b_init: Optional[snt.initializers.Initializer] = None,
#                  with_bias: bool = True,
#                  activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.relu,
#                  dropout_rate: Optional[float] = None,
#                  activate_final: bool = False,
#                  name: Optional[str] = None):
#         super().__init__(name=name)

#         if not with_bias and b_init is not None:
#             raise ValueError("b_init must be None if with_bias is False.")

#         self._activation = activation
#         self._dropout_rate = dropout_rate
#         self._with_bias = with_bias
#         self._w_init = w_init
#         self._b_init = b_init
#         self._activate_final = activate_final

#         self._layers = []
#         for i, output_size in enumerate(output_sizes):
#             self._layers.append(
#                 snt.Linear(
#                     output_size=output_size,
#                     with_bias=with_bias,
#                     w_init=w_init,
#                     b_init=b_init,
#                     name=f"linear_{i}"
#                 )
#             )

#     def __call__(self, inputs: tf.Tensor, is_training: Optional[bool] = None) -> tf.Tensor:
#         use_dropout = self._dropout_rate not in (None, 0.0)

#         if use_dropout and is_training is None:
#             raise ValueError("is_training must be specified when using dropout.")
#         elif not use_dropout and is_training is not None:
#             raise ValueError("is_training should only be specified if dropout is used.")

#         x = inputs
#         num_layers = len(self._layers)

#         for i, layer in enumerate(self._layers):
#             x = layer(x)
#             if i < (num_layers - 1) or self._activate_final:
#                 if use_dropout and is_training:
#                     x = tf.nn.dropout(x, rate=self._dropout_rate)
#                 x = self._activation(x)

#         return x

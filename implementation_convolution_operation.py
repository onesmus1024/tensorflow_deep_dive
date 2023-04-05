
import tensorflow as tf
import numpy as np

x = tf.constant(
    [[
        [[1], [2], [3], [4]],
        [[4], [3], [2], [1]],
        [[5], [6], [7], [8]],
        [[8], [7], [6], [5]]
    ]],
    dtype=tf.float32)
x_filter = tf.constant(
    [[[[0.5]], [[1]]],
     [[[0.5]], [[1]]]
     ],
    dtype=tf.float32)
x_stride = [1, 1, 1, 1]
x_padding = 'VALID'
x_conv = tf.nn.conv2d(
    input=x, filters=x_filter, strides=x_stride, padding=x_padding
)


print(x)
print("x" * 20)
print(x_filter)
print("x_filter" * 20)
print(x_conv)
print("x_conv" * 20)

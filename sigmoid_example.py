import tensorflow as tf
import numpy as np



@tf.function
def layer(x, W, b):
    # Building the graph
    h = tf.nn.sigmoid(tf.matmul(x,W) + b) # Operation to perform
    return h

x = np.array([[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]],
dtype=np.float32)


init_w = tf.initializers.RandomUniform(minval=-0.1, maxval=0.1)(shape=[10, 5])
W = tf.Variable(init_w, dtype=tf.float32, name='W')
init_b = tf.initializers.RandomUniform()(shape=[5])
b = tf.Variable(init_b, dtype=tf.float32, name='b')


h = layer(x, W, b)

print(x)
print("x" * 20 )
print(W.numpy())
print("w" * 20 )
print(b.numpy())
print("b" * 20 )
print(h)
print("h" * 20 )

print(f"h = {h.numpy()}")

import hd
import numpy as np
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

log_pitches = np.array([[7.0]]) / 12.0
vectors = hd.vectors.space_graph(3, 2)
res = sess.run(hd.scaled_hd_graph(log_pitches, vectors, c=0.05))

print(res)
print(res.shape)

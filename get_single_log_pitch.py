import hd
import numpy as np
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# log_pitches = np.array([[7.0, 10.0]]) / 12.0
log_pitches = np.array([[0.17874651], [0.35929495]])
vectors = hd.vectors.space_graph(5, 4)
# res = sess.run(hd.scaled_hd_graph(log_pitches, vectors, c=0.1))

res = sess.run(hd.vectors.sorted_from_log(log_pitches, vectors, n_returned=20))

print([hd.vectors.to_ratio(v) for v in res[0]])
print([hd.vectors.to_ratio(v) for v in res[1]])
# print(res.shape)

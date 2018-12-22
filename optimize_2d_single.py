import hd
import numpy as np
import tensorflow as tf

c = 0.2
n_points = 16
LEARNING_RATE = 1.0e-3
CONVERGENCE_THRESHOLD = 1.0e-32
MAX_ITERS = 1000000

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

dimensions = 2
batch_size = 64
log_pitches = tf.get_variable("log_pitches", [1, dimensions], dtype=tf.float64)

sess.run(tf.global_variables_initializer())

vectors = hd.vectors.space_graph(5, 4, bounds=(0.0, 1.0), name="vectors")
assign_op = log_pitches.assign([[0.23656219, 0.47333019]])

stopping_condition_op = hd.optimize.stopping_op(vectors, log_pitches, lr=LEARNING_RATE, ct=CONVERGENCE_THRESHOLD, c=c)

all_possible_pitches_log = np.empty([0, 2])

sess.run([assign_op])
for idx in range(MAX_ITERS):
    if (sess.run(stopping_condition_op)):
        print("Converged at iteration: ", idx)
        out_pitches = np.array(sess.run(log_pitches))
        all_possible_pitches_log = np.concatenate([all_possible_pitches_log, out_pitches])
        break

print(all_possible_pitches_log.shape)
log_vectors = hd.vectors.pd_graph(vectors)

print(all_possible_pitches_log)
diffs_to_poles = tf.abs(tf.tile(log_vectors[:, None, None], [1, 1, 2]) - all_possible_pitches_log)
mins = tf.argmin(diffs_to_poles, axis=0)
winner = tf.map_fn(lambda m: tf.map_fn(lambda v: vectors[v], m, dtype=tf.float64), mins, dtype=tf.float64)
winners = sess.run(winner)
print(winners)

def vector_to_ratio(vector):
    primes = hd.PRIMES[:vector.shape[0]]
    num = np.where(vector > 0, vector, np.zeros_like(primes))
    den = np.where(vector < 0, vector, np.zeros_like(primes))
    return (
        np.product(np.power(primes, num)), 
        np.product(primes ** np.abs(den))
    )

all_possible_pitches = set()

for row in winners:
    all_possible_pitches.add(tuple([vector_to_ratio(r) for r in row]))

print(len(all_possible_pitches))
print(sorted(all_possible_pitches, key=lambda r: (r[0][0] / r[0][1], r[1][0] / r[1][1])))
import tensorflow as tf

from .tenney import scaled_hd_graph

def stopping_op(vectors, log_pitches, lr=1.0e-3, ct=1.0e-16, c=0.1):
    opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
    loss = scaled_hd_graph(log_pitches, vectors, c=c)
    opt_op = opt.minimize(loss, var_list=[log_pitches])
    compute_grad_op = opt.compute_gradients(loss, var_list=[log_pitches])
    grad_norms_op = [tf.nn.l2_loss(g) for g, v in compute_grad_op]
    grad_norm_op = tf.add_n(grad_norms_op, name="grad_norm")

    with tf.control_dependencies([opt_op]):
        vals = tf.less(grad_norm_op, tf.constant(ct, dtype=tf.float64))
        stopping_condition_op = tf.reduce_all(vals)
    
    return stopping_condition_op

import tensorflow as tf

def stopping_op(loss, var_list, lr=1.0e-3, ct=1.0e-16):
    """
    Given a loss function and a variable list, gives a stopping condition Tensor that 
    can be evaluated to see whether the variables have properly converged. 
    """
    opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
    opt_op = opt.minimize(loss, var_list=var_list)
    compute_grad_op = opt.compute_gradients(loss, var_list=var_list)
    grad_norms_op = [tf.nn.l2_loss(g) for g, v in compute_grad_op]
    grad_norm_op = tf.add_n(grad_norms_op, name="grad_norm")

    with tf.control_dependencies([opt_op]):
        vals = tf.less(grad_norm_op, tf.constant(ct, dtype=tf.float64))
        stopping_condition_op = tf.reduce_all(vals)
    
    return stopping_condition_op

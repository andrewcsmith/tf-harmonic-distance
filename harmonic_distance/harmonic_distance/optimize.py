import tensorflow as tf

from .utilities import reduce_parabola

def stopping_op(loss, var_list, lr=1.0e-3, ct=1.0e-16):
    """
    Given a loss function and a variable list, gives a stopping condition Tensor that 
    can be evaluated to see whether the variables have properly converged. 
    """
    opt = tf.optimizers.Adagrad(learning_rate=lr)
    compute_grad_op = opt.get_gradients(loss, var_list)
    # grad_norms_op = [tf.nn.l2_loss(g) for g, v in compute_grad_op]
    # grad_norm_op = tf.add_n(grad_norms_op, name="grad_norm")

    # with tf.control_dependencies([opt.apply_gradients(compute_grad_op)]):
    #     vals = tf.less(grad_norm_op, tf.constant(ct, dtype=tf.float64))
    #     stopping_condition_op = tf.reduce_all(vals)
    
    # return stopping_condition_op

def parabolic_loss_function(pds, hds, log_pitches, curves=None):
    """
    pds: Pitch distance coordinate values of each vector in the space
    hds: Aggregate harmonic distance values of each vector in the space
    log_pitches: The set of pitches to evaluate
    """
    distances = tf.map_fn(lambda x: reduce_parabola(pds - x, curves=curves), log_pitches)
    return tf.reduce_min(distances * hds + hds, axis=1)

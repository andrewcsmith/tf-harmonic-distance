import tensorflow as tf

from .utilities import reduce_parabola

def stopping_op(get_loss, var_list, lr=1.0e-3, ct=1.0e-16):
    """
    Given a loss function and a variable list, gives a stopping condition Tensor that 
    can be evaluated to see whether the variables have properly converged. 
    """
    opt = tf.optimizers.Adam(learning_rate=lr)

    with tf.GradientTape() as g:
        loss = get_loss()
        dz_dv = g.gradient(loss, var_list)
        # print("Gradients: ", dz_dv)
    norms = tf.nn.l2_loss(dz_dv)
    stop = norms < ct
    opt.apply_gradients(zip(dz_dv, var_list))
    return stop

def parabolic_loss_function(pds, hds, log_pitches, curves=None):
    """
    pds: Pitch distance coordinate values of each vector in the space
    hds: Aggregate harmonic distance values of each vector in the space
    log_pitches: The set of pitches to evaluate
    """
    distances = reduce_parabola(pds[:, None] - log_pitches, axis=len(log_pitches.shape), curves=curves)
    scaled = ((2.0 ** hds)[:, None] * distances) + hds[:, None]
    return tf.reduce_min(scaled, axis=0)

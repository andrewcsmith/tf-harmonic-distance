import tensorflow as tf
import datetime

from .utilities import reduce_parabola, log2_graph
from .vectors import VectorSpace

def parabolic_loss_function(pds, hds, log_pitches, curves=None):
    """
    pds: Pitch distance coordinate values of each vector in the space
    hds: Aggregate harmonic distance values of each vector in the space
    log_pitches: The set of pitches to evaluate
    """
    distances = reduce_parabola(pds[:, None] - log_pitches, axis=len(log_pitches.shape), curves=curves)
    scaled = ((2.0 ** hds)[:, None] * distances) + hds[:, None]
    return tf.reduce_min(scaled, axis=0)

class Minimizer(tf.Module):
    def __init__(self, dimensions=1, learning_rate=1.0e-2, max_iters=1000, convergence_threshold=1.0e-5, c=0.01, batch_size=1, **kwargs):
        """
        dimensions: Dimensionality of the space. This is only finding the
        minimum of a single interval.
        learning_rate: The learning rate of the optimization algorithm. A higher
        value will converge more quickly, if possible, but might never converge.
        max_iters: Maximum number of iterations before giving up on convergence.
        convergence_threshold: The convergence threshold is the norm of the
        gradients of the loss function. This is used to test whether a proper
        "valley" has been found.
        c: The "curve" of the parabola around each possible pitch. A higher
        value will lead to fewer possible pitches.
        """
        self.dimensions = dimensions
        self.learning_rate = tf.Variable(learning_rate)
        self.max_iters = max_iters
        self.convergence_threshold = convergence_threshold
        self.curves = tf.Variable(tf.ones([dimensions], dtype=tf.float64))
        self.set_all_curves(c)
        self.vs = VectorSpace(dimensions=dimensions, **kwargs)
        self.log_pitches = tf.Variable(tf.zeros(([batch_size, dimensions]), dtype=tf.float64), dtype=tf.float64)
        self.step = tf.Variable(0, dtype=tf.int64)
        self.opt = tf.optimizers.Adadelta(learning_rate=self.learning_rate)
        self.opt.minimize(self.static_loss, self.log_pitches)

    def set_all_curves(self, c):
        self.curves.assign([c for _ in range(self.dimensions)])

    @tf.function
    def minimize(self):
        self.step.assign(0)
        self.reinitialize_weights()
        while self.stopping_op() and self.step < self.max_iters:
            self.opt.minimize(self.loss, self.log_pitches)
            self.step.assign_add(1)
    
    @tf.function
    def reinitialize_weights(self):
        for w in self.opt.weights[1:]:
            w.assign(tf.zeros_like(w))
    
    def minimize_logged(self, log=False):
        self.step.assign(0)
        self.reinitialize_weights()
        if log:
            self.writers = [self.timestamped_writer(var='/main')]
            for idx in range(self.dimensions):
                self.writers.append(self.timestamped_writer(var=("/pitch{}".format(idx+1))))
        if log:
            self.write_values()
        while self.stopping_op() and self.step < self.max_iters:
            self.opt.minimize(self.loss, self.log_pitches)
            self.step.assign_add(1)
            if log:
                self.write_values()
        if log:
            if self.stopping_op():
                with self.writers[0].as_default():
                    tf.summary.text("convergence", "did not converge", step=self.step)
            else:
                with self.writers[0].as_default():
                    tf.summary.text("convergence", "converged", step=self.step)
                
    def timestamped_writer(self, var=''):
        return tf.summary.create_file_writer('logs/fit/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + var)

    def write_values(self):
        current_loss = self.loss()
        with self.writers[0].as_default():
            tf.summary.scalar("loss", tf.reduce_mean(current_loss, axis=-1), step=self.step)
            with tf.GradientTape() as g:
                dz_dv = g.gradient(self.loss, self.log_pitches)
            norms = tf.nn.l2_loss(dz_dv)
            tf.summary.scalar("loss-norm", norms, step=self.step)
        for idx, writer in enumerate(self.writers[1:]):
            with writer.as_default():
                tf.summary.scalar("loss", current_loss[idx], step=self.step)
                tf.summary.scalar("pitch-cents", self.log_pitches[idx] * 1200.0, step=self.step)

    @tf.function
    def loss(self):
        return parabolic_loss_function(self.vs.pds, self.vs.hds, self.log_pitches, curves=self.curves)
            
    @tf.function
    def stopping_op(self):
        """
        Given a loss function and a variable list, gives a stopping condition
        Tensor that can be evaluated to see whether the variables have properly
        converged. 
        """
        with tf.GradientTape() as g:
            dz_dv = g.gradient(self.loss, self.log_pitches)
        norms = tf.nn.l2_loss(dz_dv)
        return norms >= self.convergence_threshold
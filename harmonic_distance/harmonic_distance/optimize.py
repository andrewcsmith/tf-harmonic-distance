import tensorflow as tf
import datetime

from .utilities import reduce_parabola, log2_graph, transform_to_unit_circle, transform_from_unit_circle
from .vectors import VectorSpace

@tf.function
def parabolic_loss_function(pds, hds, log_pitches, curves=None):
    """
    pds: Pitch distance coordinate values of each vector in the space
    hds: Aggregate harmonic distance values of each vector in the space
    log_pitches: The set of pitches to evaluate
    """
    distances = reduce_parabola(pds[:, None] - log_pitches, axis=-1, curves=curves)
    scaled = ((2.0 ** hds)[:, None] * distances) + hds[:, None]
    return tf.reduce_min(scaled, axis=0)

class Minimizer(tf.Module):
    def __init__(self, dimensions=1, learning_rate=1.0e-2, max_iters=1000, convergence_threshold=1.0e-5, c=0.01, batch_size=1, callbacks=None, name=None, vs=None, **kwargs):
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
        super().__init__(name=name)
        self.dimensions = dimensions
        self.learning_rate = learning_rate # tf.Variable(learning_rate)
        self.max_iters = max_iters
        self.convergence_threshold = tf.Variable(
            float(convergence_threshold),
            trainable=False,
            dtype=tf.float64,
            name="convergence_threshold",
        )
        self.curves = tf.Variable(tf.ones([dimensions], dtype=tf.float64), trainable=False)
        self.set_all_curves(c)
        self.vs = vs if vs is not None else VectorSpace(dimensions=dimensions, **kwargs)
        self.log_pitches = tf.Variable(tf.zeros([batch_size, dimensions], dtype=tf.float64), dtype=tf.float64, name="log_pitches")
        self.active_mask = tf.Variable(tf.ones([batch_size], dtype=tf.float64), trainable=False, name="active_mask")
        self.step = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.opt = tf.optimizers.Adadelta(learning_rate=self.learning_rate)
        self.opt.build([self.log_pitches])
        self.snapshot_optimizer_state()
        self.callbacks = tf.keras.callbacks.CallbackList(callbacks=callbacks)
        self.callbacks.set_model(self)

    def set_all_curves(self, c):
        self.curves.assign([c for _ in range(self.dimensions)])

    def set_convergence_threshold(self, convergence_threshold):
        convergence_threshold = float(convergence_threshold)
        if convergence_threshold <= 0.0:
            raise ValueError(f"convergence_threshold must be positive; got {convergence_threshold}")
        self.convergence_threshold.assign(convergence_threshold)

    def set_active_count(self, active_count):
        batch_size = self.log_pitches.shape[0]
        if active_count < 0 or active_count > batch_size:
            raise ValueError(f"active_count must be between 0 and {batch_size}; got {active_count}")
        mask = tf.concat([
            tf.ones([active_count], dtype=tf.float64),
            tf.zeros([batch_size - active_count], dtype=tf.float64),
        ], axis=0)
        self.set_active_mask(mask)

    def real_log_pitches(self):
        """
        The current log_pitches in real (untransformed) log2-ratio coordinates.
        When the vector space is polar, log_pitches live in the transformed
        space and are inverse-transformed here; otherwise returned as-is.
        """
        if getattr(self.vs, "polar", False):
            return transform_from_unit_circle(self.log_pitches)
        return tf.identity(self.log_pitches)

    def set_real_log_pitches(self, log_pitches):
        """
        Assign log_pitches given in real log2-ratio coordinates, transforming
        into the polar space when the vector space is polar.
        """
        log_pitches = tf.convert_to_tensor(log_pitches, dtype=tf.float64)
        if getattr(self.vs, "polar", False):
            log_pitches = transform_to_unit_circle(log_pitches)
        self.log_pitches.assign(log_pitches)

    def set_active_mask(self, active_mask):
        mask = tf.convert_to_tensor(active_mask, dtype=tf.float64)
        batch_size = self.log_pitches.shape[0]
        if mask.shape.ndims != 1:
            raise ValueError(f"active_mask must be one-dimensional; got shape {mask.shape}")
        if mask.shape[0] != batch_size:
            raise ValueError(f"active_mask must have length {batch_size}; got {mask.shape[0]}")
        mask = tf.where(mask != 0.0, tf.ones_like(mask), tf.zeros_like(mask))
        self.active_mask.assign(mask)
    
    @tf.function
    def opt_minimize(self):
        with tf.GradientTape(persistent=False) as g:
            dz_dv = g.gradient(self.loss(), self.log_pitches)
        dz_dv = dz_dv * self.active_mask[:, None]
        self.opt.apply([dz_dv])

    @tf.function
    def minimize(self):
        self.step.assign(0)
        self.reinitialize_optimizer_state()
        while self.stopping_op() and self.step < self.max_iters:
            self.opt_minimize()
            self.step.assign_add(1)
            self.callbacks.on_train_batch_end(self.step)

    def snapshot_optimizer_state(self):
        self.initial_optimizer_state = [
            tf.Variable(tf.identity(var), trainable=False)
            for var in self.opt.variables
        ]

    @tf.function
    def reinitialize_optimizer_state(self):
        for var, initial_value in zip(self.opt.variables, self.initial_optimizer_state):
            var.assign(initial_value)

    @tf.function
    def reinitialize_weights(self):
        self.reinitialize_optimizer_state()
    
    def minimize_logged(self, log=False):
        self.step.assign(0)
        self.reinitialize_optimizer_state()
        if log:
            self.writers = [self.timestamped_writer(var='/main')]
            for idx in range(self.dimensions):
                self.writers.append(self.timestamped_writer(var=("/pitch{}".format(idx+1))))
        if log:
            self.write_values()
        while self.stopping_op() and self.step < self.max_iters:
            self.opt_minimize()
            self.step.assign_add(1)
            self.callbacks.on_train_batch_end(self.step)
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
        if hasattr(self.vs, "loss"):
            return self.vs.loss(self.log_pitches, curves=self.curves)
        return parabolic_loss_function(self.vs.pds, self.vs.hds, self.log_pitches, curves=self.curves)
            
    @tf.function
    def stopping_op(self):
        """
        Given a loss function and a variable list, gives a stopping condition
        Tensor that can be evaluated to see whether the variables have properly
        converged. 
        """
        with tf.GradientTape() as g:
            dz_dv = g.gradient(self.loss(), self.log_pitches)
        dz_dv = dz_dv * self.active_mask[:, None]
        norms = tf.nn.l2_loss(dz_dv)
        return norms >= self.convergence_threshold

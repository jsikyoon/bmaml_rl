

from sandbox.rocky.tf.algos.bmaml_npo import BMAMLNPO
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer


class BMAMLTRPO(BMAMLNPO):
    """
    Trust Region Policy Optimization
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            **kwargs):
        assert optimizer is None
        assert optimizer_args is None
        n_particles = len(kwargs['policy_list'])
        optimizer_list = []
        for n in range(n_particles):
            optimizer_list.append(ConjugateGradientOptimizer(**dict()))
        super(BMAMLTRPO, self).__init__(optimizer_list=optimizer_list, **kwargs)

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import special_ortho_group


class RotationProblem:
    """
        An implementation of "Wen et.al,. A feasible method for optimization with orthogonality constraints.",
        which is designed for optimization problems with SO(n) constraint.
    """

    def __init__(self, obj_func, grad_func, dim=3, tolerance=1e-8):
        self.obj_func = obj_func
        self.grad_func = grad_func
        self.identity = np.identity(dim)
        self.tol = tolerance
        self.dim = dim

    def _update_one_step(self, grad, R, tau):
        W = grad @ R.T - R @ grad.T
        return np.linalg.inv(self.identity + tau / 2 * W) @ (self.identity - tau / 2 * W) @ R

    def _mono_line_search(self, grad, R, tau, rho, beta):
        l_bound = self.obj_func(R) - beta * tau * 0.5 * np.linalg.norm(grad @ R.T - R @ grad.T)
        while True:
            if tau >= self.tol ** 2 and self.obj_func(self._update_one_step(grad, R, tau)) >= l_bound:
                tau *= rho
            else:
                break
        return max(tau, self.tol ** 2)

    def _optimize(self, init_R, max_iter=1000, init_tau=1e-2, beta=0.01, rho=0.5):
        R, prev_R = init_R, [init_R]
        for _ in range(max_iter):
            grad = self.grad_func(R)
            tau = self._mono_line_search(grad=grad, R=R, tau=init_tau, rho=rho, beta=beta)
            R = self._update_one_step(grad=grad, R=R, tau=tau)
            prev_R.append(R)
            if np.linalg.norm(prev_R[-1] - prev_R[-2]) <= self.tol:
                break
        return R, self.obj_func(R)

    def solve(self, max_iter=1000, num_start=180, all_init_R=None, init_tau=1, beta=1e-4, rho=0.5, n_thread=1):
        """
        :param max_iter  : maximum iterations for one start
        :param num_start : number of initializations
        :param all_init_R: a list of all initializations, if None, num_start samples are drawn from the Haar distribution
        :param init_tau  : initialization step size, can be large
        :param beta      : Armijo rule for line search, should be small enough
        :param rho       : line search decay parameter
        :param n_thread  : number of parallel threads
        :return          : estimated R, associated energy
        """

        ## check parameters
        assert beta < 0.5, 'Armijo rule is only applicable for beta less than 0.5'
        assert rho < 1, 'Armijo rule requires the decay factor rho to be less than 1'
        ## solve
        all_init_R = special_ortho_group.rvs(self.dim, num_start) if all_init_R is None else all_init_R
        res = Parallel(n_jobs=min(n_thread, len(all_init_R)))(delayed(self._optimize)
                                                              (_, max_iter, init_tau, beta, rho) for _ in all_init_R)
        return min(res, key=lambda x: x[1])


from RotationProblem import RotationProblem
import numpy as np

def solve_problem(A, B, C):
    """
        Example: use the solver to solve the following SO(3)-constrained problem"
                 min || C - A @ R @ B @ R.T ||_F^2,  s.t.  R in SO(3)
    """

    def _obj(_R):
        """ Objective function, input is a SO(n) matrix and output is a scalar """
        return 0.5 * np.linalg.norm(C - A @ _R @ B @ _R.T) ** 2

    def _grad(_R):
        """ Gradient function, input is a SO(n) matrix and output is the same-shaped Jacobian """
        tmp = A @ _R @ B @ _R.T - C
        return A.T @ tmp @ _R @ B + tmp.T @ A @ _R @ B

    # call the solver
    esti_R, esti_energy = RotationProblem(obj_func=_obj, grad_func=_grad, dim=3).solve(n_thread=16, num_start=1000)
    return esti_R, esti_energy


if __name__ == '__main__':

    # generate data
    from scipy.stats import special_ortho_group
    A, B = np.random.rand(3, 3), np.random.rand(3, 3)
    R_gt = special_ortho_group.rvs(3)
    C_gt = A @ R_gt @ B @ R_gt.T

    # optimizing...
    esti_R, esti_energy = solve_problem(A=A, B=B, C=C_gt)

    print('esti_R:\n', esti_R)
    print('GT_R:\n', R_gt)
    print('final energy:', esti_energy)
    print('err to GT:', np.linalg.norm(esti_R - R_gt))


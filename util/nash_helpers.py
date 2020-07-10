import gambit
import numpy as np
from scipy import optimize as op


def find_zero_sum_mixed_ne(payoff):
    """
    Function for returning mixed strategies of the first step of double oracle iterations.
    :param payoff: Two dimensinal array. Payoff matrix of the players.
    The row is defender and column is attcker. This is the payoff for row player.
    :return: List, mixed strategy of the attacker and defender at NE by solving maxmini problem.
    """
    # This implementation is based on page 88 of the book multiagent systems (Shoham etc.)
    # http://www.masfoundations.org/mas.pdf
    # n_action = payoff.shape[0]
    m, n = payoff.shape
    c = np.zeros(n)
    c = np.append(c, 1)
    A_ub = np.concatenate((payoff, np.full((m, 1), -1)), axis=1)
    b_ub = np.zeros(m)
    A_eq = np.full(n, 1)
    A_eq = np.append(A_eq, 0)
    A_eq = np.expand_dims(A_eq, axis=0)
    b_eq = np.ones(1)
    bound = ()
    for i in range(n):
        bound += ((0, None),)
    bound += ((None, None),)

    res_attacker = op.linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=bound, method="interior-point")

    return res_attacker.x[0:n]


def find_general_sum_mixed_ne(attacker_payoff, defender_payoff):
    # assert attacker_payoff.shape == defender_payoff.shape

    g = gambit.Game.from_arrays((attacker_payoff * 10).astype('int').astype(gambit.Decimal),
                                (defender_payoff * 10).astype('int').astype(gambit.Decimal))
    solver = gambit.nash.ExternalGlobalNewtonSolver()
    profile = solver.solve(g)

    return np.array([p for p in profile[-1]])[:attacker_payoff.shape[0]],\
           np.array([p for p in profile[-1]])[attacker_payoff.shape[0]:]
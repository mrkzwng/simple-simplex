import numpy as np
import math
import sys
import os
from itertools import starmap
from test import *

'''
NOTE:
np.isclose replaced with math.isclose
- keep an eye on abs_tol, the upper-
  bound on differences between the
  compared values
'''


class Solver:
    '''
    The two-phase simplex algorithm via Bland's rule
    
    Based on Bertsimas & TsiTsiklis's text
    "Introduction to Linear Optimization",
    pg. 99-128, section "Naive Implementation",
    but with more redundant calculations 
    '''
    def __init__(self, A, b, c, x_constraints, Ax_constraints, problem, abs_tol=1e-9):
        '''
        transforms LP into standard form
        - checks constraints' independence
        - handles unconstrained variables
        - adds slack variables

        expects input LP in the form of:
            min    c'x
            s.t. Ax <= b
        x_i in x may or may not be constrained
        '''
        # constraints coefficients
        self.A = A
        # constraints constants
        self.b = b
        # sign constraints on variables
        self.x_constraints = x_constraints
        # objective coefficients
        self.c = c
        # linear system constraints
        self.Ax_constraints = Ax_constraints
        # basic coefficients' indices
        self.basic_idxs = None
        # basic feasible solutions
        self.x = None
        # dictionary for keeping track of replaced vars
        self.x_dict = None
        # numerical error tolerance
        self.abs_tol=abs_tol
        # to only remove unconstrained vars and add slacks once
        self.constraints_added = False
        self.slacks_added = False
        # to flip optimal results when minimization is complete
        self.problem = problem

        if not self._is_independent(self.A):
            self._terminate('Dependent constraints. Exiting.')


    def dual(self):
        '''
        prints the dual LP
        '''
        self._add_constraints(self.A, self.c, self.x_constraints)
        self._add_slacks(self.A, self.Ax_constraints, self.c)
        self._gen_dual_x()
        self._gen_dual_A()
        self._gen_dual_b()
        self._gen_dual_c()
        self._print_dual_x()
        self._print_dual_A()
        self._print_dual_b()
        self._print_dual_c()


    def solve(self):
        '''
        tries to solve the LP;
        exits if not bounded feasible
        '''
        self._add_constraints(self.A, self.c, self.x_constraints)
        self._add_slacks(self.A, self.Ax_constraints, self.c)
        self._phase_one()
        interim_solution = self._phase_two()
        optimal_val = np.matmul(self.c, self.x)
        optimal_val = optimal_val if self.problem == 'min' else -optimal_val
        solution = self._recover_solution(interim_solution,
                                          self.x_dict, 
                                          self.x_constraints)
        sol_str = '\t'.join([str(float(s)) for s in solution])

        print(float(optimal_val))
        print(sol_str)

    '''
    in 'dual()'
    '''
    def _gen_dual_x(self):
        '''
        generate unconstrained dual x
        '''
        self.dual_x_constraints = np.zeros(self.A.shape[0])


    def _gen_dual_A(self):
        '''
        generate dual A = A transpose
        '''
        self.dual_A = self.A.transpose()


    def _gen_dual_b(self):
        '''
        generate dual b = c, with dual Ax <= b
        constraints
        '''
        self.dual_b = self.c
        self.dual_Ax_constraints = np.ones(self.c.shape[0]) * (-1)


    def _gen_dual_c(self):
        '''
        generate dual c = b, with a
        maximization objective
        '''
        self.dual_c = self.b 


    def _print_dual_x(self):

        print('### x_FILE ###')
        row_str = '\t'.join([constraint(cons_i) 
                             for cons_i in self.dual_x_constraints])
        print(row_str)


    def _print_dual_A(self):

        print('### A_FILE ###')
        for row_idx, row in enumerate(self.dual_A):
            row_str = '\t'.join([sign(e)+str(np.abs(float(e))) for e in row])
            print(row_str)


    def _print_dual_b(self):

        print('### b_FILE ###')
        for idx, (cons_i, b_i) in enumerate(zip(self.dual_Ax_constraints, 
                                                self.dual_b)):
            row_str = constraint(cons_i)+'\t'+sign(b_i)+str(np.abs(float(b_i)))
            print(row_str)


    def _print_dual_c(self):

        print('### c_FILE ###')
        row_str = '\t'.join([sign(c_i)+str(np.abs(float(c_i))) 
                             for c_i in self.dual_c])
        print('max')
        print(row_str)

    '''
    in 'solve'
    '''
    def _add_constraints(self, A, c, x_constraints):
        '''
        replaces each unconstrained variable with
        two constrained variables >= zero; changes
        non-positive constraints to non-negative

        the replacement is represented as a dictionary
        of indices:
            d = {i: [j, j+1] | all unconstrained x_i,
                 i: [k] | all constrained x_i}
        where i is the index of the original vector of
        variables, and [j, j+1], k are the indices of the
        split positive and negative parts and unsplit
        variable, respectively 

        consistent adjustments to A and c are made
        '''
        if self.constraints_added:
            return

        x_dict = {}
        j = 0
        for orig_idx, x_cons_i in enumerate(x_constraints):
            # if we have an unconstrained variable
            if math.isclose(x_cons_i, 0, abs_tol=self.abs_tol):
                x_dict[orig_idx] = [j, j+1]
                # we want to duplicate the constraints for the var
                dupl_idxs = [idx for idx in range(j+1)]
                dupl_idxs.extend([idx for idx in range(j, A.shape[1])])
                # and flip the duplicate constraints' column's sign
                A = A[:, dupl_idxs]
                A[:, j+1] = (-1) * A[:, j+1]
                c = c[dupl_idxs]
                c[j+1] = (-1) * c[j+1]
                j = j + 2
            # if our variable is non-positive
            elif x_cons_i < 0:
                # flip constraint signs; simplex treats it as positive
                x_dict[orig_idx] = j
                A[:, j] = (-1) * A[:, j]
                c[j] = (-1) * c[j]
                j = j + 1
            else:
                x_dict[orig_idx] = j
                j = j + 1

        self.A = A
        self.c = c
        self.x_dict = x_dict
        self.constraints_added = True


    def _add_slacks(self, A, Ax_constraints, c):
        '''
        add slack variables via extending A

        each e_i in Ax_constraints is:
            1: if an <= constraint
            0: if an equality constraint
        '''
        if self.slacks_added:
            return

        I = np.eye(Ax_constraints.shape[0])
        slack_idxs = [i for i in range(Ax_constraints.shape[0])
                      if Ax_constraints[i] == 1]
        A_slack = I[:, slack_idxs]
        c_slack = np.zeros(A_slack.shape[1])
        A = np.concatenate([A, A_slack], axis=1)
        c = np.concatenate([c, c_slack])

        self.c = c
        self.A = A
        self.slacks_added = True


    def _phase_one(self):
        '''
        'phase one' of the simplex
        - creates artificial BFS, auxiliary LP
        - solves auxiliary LP via artificial BFS
        - handles degenerate auxiliary optimal BFS

        - checks for feasibility of original LP
        '''
        self._set_nonnegative_b(self.A, self.b)
        self._create_auxiliary(self.A, self.b)

        optimal = False
        while not optimal:
            optimal, aux_x = self._next_solution(self.aux_A,
                                                 self.aux_c,
                                                 self.aux_x,
                                                 self.aux_basic_idxs)
            self.aux_x = aux_x

        if self._is_infeasible(self.aux_c, self.aux_x):
            self._terminate('Infeasible problem. Exiting.')
        if self.aux_basic_idxs[-1] > self.A.shape[0] - 1:
            self._driveout_artificial(self.aux_basic_idxs, self.aux_A)

        self.basic_idxs = self.aux_basic_idxs
        self.x = self.aux_x[:self.A.shape[1]]
        del (self.aux_A, self.aux_c, self.aux_x, self.aux_basic_idxs)


    def _phase_two(self):
        '''
        'phase two' of the simplex
        - finds optimal BFS via Bland's rule
        - finds entering variable 
        - finds exiting variable

        - checks for optimality of LP
        - checks for boundedness of LP

        returns results when optimal
        '''
        optimal = False
        while not optimal:
            optimal, x = self._next_solution(self.A,
                                             self.c,
                                             self.x,
                                             self.basic_idxs)
            self.x = x

        return(self.x)


    def _recover_solution(self, x, x_dict, x_constraints):
        '''
        recover solution from constraints fudging,
        sign flipping, slacks, etc. in _add_constraints,
        _add_slacks

        returns values of variables in the original
        objective function
        '''
        recovered_x = []
        for orig_idx in range(len(x_constraints)):
            # if we have an unconstrained variable split
            if type(x_dict[orig_idx]) == list:
                new_idxs = x_dict[orig_idx]
                # if both values are zero, the var is zero
                if math.isclose(sum(x[new_idxs]), 0, abs_tol=self.abs_tol):
                    recovered_x.append(0)
                # if positive part is non-zero, var is positive
                elif x[new_idxs[0]] > 0:
                    recovered_x.append(x[new_idxs[0]])
                # otherwise, it's negative
                else:
                    recovered_x.append(x[new_idxs[1]] * (-1))
            # if the variable was originally non-positive
            elif x_constraints[orig_idx] < 0:
                # flip its sign back
                new_idx = x_dict[orig_idx]
                recovered_x.append(x[new_idx] * (-1))
            # otherwise, everything is fine
            else:
                new_idx = x_dict[orig_idx]
                recovered_x.append(x[new_idx])

        return(recovered_x)

    '''
    Phase 1
    '''
    def _set_nonnegative_b(self, A, b):
        '''
        sets RHS constant bounds b for constraints A
        to be non-negative so that artificial vars
        can all be non-negative
        '''
        b_sign = np.sign(b)
        b_sign[b_sign == 0] = 1
        self.A = b_sign.reshape([b.shape[0], 1]) * A
        self.b = b_sign * b


    def _create_auxiliary(self, A, b):
        '''
        creates auxiliary LP with initial
        BFS aux_x = [0 ... 0 b]
        '''
        self.aux_A = np.concatenate([A, np.eye(A.shape[0])], axis=1)
        self.aux_c = np.concatenate([np.zeros(A.shape[1]), 
                                     np.ones(A.shape[0])])
        self.aux_basic_idxs = [idx for idx 
                               in range(A.shape[1], self.aux_A.shape[1])]
        self.aux_x = np.concatenate([np.zeros(A.shape[1]), b])
        

    def _driveout_artificial(self, aux_basic_idxs, aux_A):
        '''
        drives out artificial variables
        from the auxiliary BFS when the
        auxiliary solution is optimal
        but degenerate
        '''
        aux_A_basic = aux_A[:, aux_basic_idxs]
        aux_A_basic_inv = np.linalg.inv(aux_A_basic)
        M = np.matmul(aux_A_basic_inv, aux_A)
        idx_dict = {}

        for var_idx, art_idx in enumerate(aux_basic_idxs):
            if art_idx > M.shape[0] - 1:
                row_idx = var_idx
                col_idx = min([idx for idx, val in enumerate(M[row_idx, :])
                               if val != 0])
                idx_dict[art_idx] = col_idx

        # side effects
        for art_idx in idx_dict.keys():
            aux_basic_idxs.remove(art_idx)
            aux_basic_idxs.append(idx_dict[art_idx])
            aux_basic_idxs.sort()

    '''
    Phase 2
    '''
    def _next_solution(self, A, c, x, basic_idxs):
        '''
        performs one iteration of the simplex
        '''
        reduced_cost = self._calc_reduced_cost(A, c, basic_idxs)

        if self._is_optimal(reduced_cost):
            return(True, x)

        entering_idx = self._calc_entering_var(reduced_cost)
        d = self._calc_direction(entering_idx, A, basic_idxs)

        if self._is_unbounded(d):
            self._terminate('Unbounded problem. Exiting.')

        exiting_idx, theta = self._calc_exiting_var(d, x)

        x = x + theta*d
        # side effects
        basic_idxs.remove(exiting_idx)
        basic_idxs.append(entering_idx)
        basic_idxs.sort()

        return(False, x)


    def _calc_reduced_cost(self, A, c, basic_idxs):
        '''
        calculates reduced cost
        '''
        c_basic = c[basic_idxs]
        A_basic_inv = np.linalg.inv(A[:, basic_idxs])
        reduced_cost = c - np.matmul(c_basic, 
                                     np.matmul(A_basic_inv, A))

        return(reduced_cost)


    def _calc_entering_var(self, reduced_cost):
        '''
        finds entering variable via smallest 
        subscript rule

        returns entering_idx, index of the entering
        variable

        NOTE:
        Will have incorrect indexing if all reduced cost
        are non-negative.

        Run _is_optimal first.
        '''

        return(min([i for i, cost_i in enumerate(reduced_cost) 
                    if cost_i < 0]))


    def _calc_direction(self, entering_idx, A, basic_idxs):
        '''
        calculates feasible direction d of 
        adjacent BFS
        '''
        A_basic_inv = np.linalg.inv(A[:, basic_idxs])
        d_basic = (-1) * np.matmul(A_basic_inv, A[:, entering_idx])
        d = np.zeros(A.shape[1])
        d[basic_idxs] = d_basic
        d[entering_idx] = 1

        return(d)


    def _calc_exiting_var(self, d, x):
        '''
        calculates (x_i / d_i) to find optimal
        magnitude along adjacent direction d to move;

        exiting_idx:
            index of the exiting variable
        theta:
            value of the minimal -(x_i / d_i)

        returns exiting_idx, theta
        '''
        thetas = list(starmap(lambda x_i, d_i: (-1) * float(x_i) / d_i 
                         if d_i < 0 else sys.maxsize, zip(x, d)))
        exiting_idx, theta = [(idx, theta_i) for idx, theta_i 
                              in enumerate(thetas) 
                              if theta_i == min(thetas)][0]

        return(exiting_idx, theta)

    '''
    termination checks
    '''
    def _is_independent(self, A):
        '''
        calculates independence of constraints via
        checking A's rank
        '''
        rank = np.linalg.matrix_rank(A)

        return(rank == A.shape[0])


    def _is_infeasible(self, aux_c, aux_x):
        '''
        checks for feasibility in phase one;
        the original LP is infeasible if the 
        artificial variables are non-zero
        '''
        aux_opt_val = np.matmul(aux_c, aux_x)
        roughly_zero = math.isclose(aux_opt_val, 0, abs_tol=self.abs_tol)

        return(not roughly_zero)


    def _is_unbounded(self, d):
        '''
        checks if all d_i in d >= 0 for 
        boundedness
        '''
        roughly_geq_zero = map(lambda d_i: 
                               math.isclose(d_i, 0, abs_tol=self.abs_tol) 
                               or d_i > 0, d)

        return(all(roughly_geq_zero))


    def _is_optimal(self, reduced_cost):
        '''
        checks for optimality;
        if reduced cost contains no negative elements, 
        the current solution is optimal
        '''

        return(all(reduced_cost >= 0))


    def _terminate(self, message):
        '''
        exits program when LP is infeasible, unbounded,
        or constraints are dependent
        '''
        print(message)
        os._exit(0)


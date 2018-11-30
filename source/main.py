import sys
import os
import numpy as np
from input import *
from simplex import Solver


def main(args):

    command = args[1]

    is_valid_command(command)

    data = get_all_data(paths=args[2:])
    solver = Solver(A=data['A'], b=data['b'], c=data['c'],
    				x_constraints=data['x'],
    				Ax_constraints=data['Ax_constraints'],
    				problem=data['problem'],
    				abs_tol=1e-15)

    if command == 'dual':
    	solver.dual()
    elif command == 'solve':
    	solver.solve()
    os._exit(0)


if __name__ == '__main__':
    main(sys.argv)
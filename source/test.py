import numpy as np

'''
helpers
'''
def sign(scalar):
    if scalar > 0:
        return('+')
    else:
        return('-')


def constraint(scalar):
    if scalar == -1:
        return('<')
    elif scalar == 1:
        return('>')
    else:
        return('=')

'''
test functions
'''
def generate_testfile(arrays, path, filetype):
    '''
    places test LP file into path
    '''
    funcs = {'A': generate_A,
             'b': generate_b,
             'c': generate_c,
             'x': generate_x}
    if filetype == 'b':
        funcs[filetype](arrays[0], arrays[1], path)
    else:
        funcs[filetype](arrays, path)


def generate_A(A, path):
    
    with open(path, 'w') as f:
        for row_idx, row in enumerate(A):
            row_str = '\t'.join([sign(e)+str(e) for e in row])
            
            if row_idx != A.shape[0] - 1:
                f.write(row_str + '\n')
            else:
                f.write(row_str)


def generate_b(b, Ax_constraints, path):

    with open(path, 'w') as f:
        for idx, (cons_i, b_i) in enumerate(zip(Ax_constraints, b)):
            row_str = constraint(cons_i)+'\t'+sign(b_i)+str(b_i)

            if idx != b.shape[0] - 1:
                f.write(row_str + '\n')
            else:
                f.write(row_str)


def generate_x(x_constraints, path):

    with open(path, 'w') as f:
        row_str = '\t'.join([constraint(cons_i) 
                             for cons_i in x_constraints])
        f.write(row_str)

def generate_c(c, path):

    with open(path, 'w') as f:
        row_str = '\t'.join([sign(c_i)+str(c_i) for c_i in c])

        f.write('min'+'\n')
        f.write(row_str)


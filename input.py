import numpy as np
import os
import sys


'''
helpers
'''
def is_number(num):
    try:
        float(num)
        return(True)
    except:
        return(False)


def translate(char):
    if char == '<': 
        return(-1)
    elif char == '>':
        return(1)
    else:
        return(0)

'''
checks file formatting
'''
def wellformed_x(x_FILE):
    odd_chars = {'<', '>', '='}
    even_char = {'\t'}
    allowed_chars = [odd_chars, even_char]
    with open(x_FILE, 'r') as f:
        contents = f.read().split('\n')

    for line_idx, line in enumerate(contents):
        if line_idx == 1:
            return(False)
        for idx, char in enumerate(line):
            if char not in allowed_chars[idx % 2]:
                return(False)

    return(True)


def wellformed_A(A_FILE):
    char_0 = lambda char: char in {'+', '-'}
    char_1 = lambda char: is_number(char)
    char_2 = lambda char: char == '\t'
    allowed_chars = [char_0, char_1, char_2]
    with open(A_FILE, 'r') as f:
        contents = f.read().split('\n')

    for line in contents:
        token_idx = 0
        idx = 0
        while idx < len(line):
            if token_idx % 3 == 1:
                j = idx
                while j < len(line) and (is_number(line[j]) 
                                         or line[j] == '.'): j += 1
                if not allowed_chars[token_idx % 3](line[idx:j]):
                    return(False)
                idx = j - 1
            elif not allowed_chars[token_idx % 3](line[idx]):
                return(False)
            token_idx += 1
            idx += 1

    return(True)


def wellformed_b(b_FILE):
    char_0 = lambda char: char in {'<', '>', '='}
    char_1 = lambda char: char == '\t'
    char_2 = lambda char: char in {'+', '-'}
    char_3 = lambda string: is_number(string)
    allowed_chars = [char_0, char_1, char_2, char_3]
    with open(b_FILE, 'r') as f:
        contents = f.read().split('\n')

    for line in contents:
        token_idx = 0
        idx = 0
        while idx < len(line):
            if token_idx % 4 == 3:
                if not allowed_chars[token_idx % 4](line[idx:]):
                    return(False)
                else:
                    break
            elif not allowed_chars[token_idx % 4](line[idx]):
                return(False)
            token_idx += 1
            idx += 1
    
    return(True)


def wellformed_c(c_FILE):
    char_0 = lambda char: char in {'+', '-'}
    char_1 = lambda char: is_number(char)
    char_2 = lambda char: char == '\t'
    allowed_chars = [char_0, char_1, char_2]
    with open(c_FILE, 'r') as f:
        contents = f.read().split('\n')

    for line_idx, line in enumerate(contents):
        if line_idx == 0 and line not in {'max', 'min'}:
            return(False)
        token_idx = 0
        idx = 0
        while line_idx > 0 and idx < len(line):
            if token_idx % 3 == 1:
                j = idx
                while j < len(line) and (is_number(line[j])
                                         or line[j] == '.'): j += 1
                if not is_number(line[idx:j]):
                    return(False)
                idx = j - 1
            elif not allowed_chars[token_idx % 3](line[idx]):
                return(False)
            token_idx += 1
            idx += 1

    return(True)

'''
parses
'''
def parse_x(x_FILE):
    '''
    parses into vector of -1, 0, 1s:
        (-1): <= 0
        (0) : unconstrained
        (1) : >= 0

    returns np.array
    '''
    with open(x_FILE, 'r') as f:
        contents = f.read().split('\n')[0]
        x_constraints = list(map(translate, contents.split('\t')))

    return(np.array(x_constraints))


def parse_A(A_FILE):
    '''
    parses 'A_FILE'

    returns np.array
    '''
    A_vals = []
    A_signs = []
    sign = lambda char: 1 if char == '+' else -1
    with open(A_FILE, 'r') as f:
        contents = f.read().split('\n')

    for line in contents:
        if len(line) == 0:
            break
        tokens = line.split('\t')
        signs = [sign(token[0]) for token in tokens]
        vals = [float(token[1:]) for token in tokens]
        A_signs.append(signs)
        A_vals.append(vals)

    A = np.array(A_signs) * np.array(A_vals)

    return(A)


def parse_b(b_FILE):
    '''
    parses 'b_FILE'

    returns flips, Ax_constraints, b
    '''
    sign = lambda char: -1 if char == '-' else 1
    with open(b_FILE, 'r') as f:
        contents = f.read().split('\n')

    Ax_constraints = np.array([translate(line[0]) for line in contents
                               if len(line) > 0])
    b_signs = np.array([sign(line[2]) for line in contents
                        if len(line) > 0])
    b_vals = np.array([float(line[3:]) for line in contents
                       if len(line) > 0])
    b = b_signs * b_vals

    return(b, Ax_constraints)


def parse_c(c_FILE):
    '''
    parses 'c_FILE'

    returns np.array

    TODO:
    Test sign flip from 'max' to 'min'
    '''
    sign = lambda char: 1 if char == '+' else -1
    with open(c_FILE, 'r') as f:
        contents = f.read().split('\n')
        opt_sign = 1 if 'min' in contents[0] else -1
        contents = contents[1]

    signs = np.array([sign(token[0]) for token in contents.split('\t')])
    values = np.array([float(token[1:]) for token in contents.split('\t')])
    c = signs * values * opt_sign
    problem = 'min' if opt_sign == 1 else 'max'

    return(c, problem)

'''
call prior to ingestion into Solver
'''
def flip_constraints(A, b, Ax_constraints):
    '''
    change all >= constraints to <=
    '''
    flip_sign = lambda sign: -1 if sign == 1 else 1
    flips = np.array([flip_sign(cons_i) for cons_i
                      in Ax_constraints])
    A = flips.reshape([flips.shape[0], 1]) * A 
    b = flips * b 
    Ax_constraints = flips * Ax_constraints * (-1)

    return(A, b, Ax_constraints)

'''
validity checks
'''
def is_valid_command(command):
    '''
    verify whether command is valid
    '''
    commands = {'dual', 'solve'}

    if command in commands:
        return(True)
    else:
        print('Unknown command. Exiting.')
        os._exit(0)


def check_valid_LP(A, b, Ax_constraints, x, c):
    '''
    checks whether coefficients' and 
    variables' dimensions are consistent
    '''
    def invalid_LP():
        print('Invalid LP. Exiting.')
        os._exit(0)

    if A.shape[0] != b.shape[0]:
        invalid_LP()
    if A.shape[1] != x.shape[0]:
        invalid_LP()
    if c.shape[0] != x.shape[0]:
        invalid_LP()
    if A.shape[0] != Ax_constraints.shape[0]:
        invalid_LP()


def grab_data(file, file_type):
    '''
    grabs data from files, checks formatting
    '''
    is_wellformed = {'A': wellformed_A,
                     'x': wellformed_x,
                     'b': wellformed_b,
                     'c': wellformed_c}
    parse = {'A': parse_A,
             'x': parse_x,
             'c': parse_c,
             'b': parse_b}
    try:
        if not is_wellformed[file_type](file):
            raise
        data = parse[file_type](file)
        return(data)
    except:
        print('Malformed input file. Exiting.')
        os._exit(0)

'''
for testing
'''
def get_all_data(paths, names=['x', 'A', 'b', 'c']):
    '''
    grabs generated input data
    '''
    data = {}
    for path, name in zip(paths, names):
        data[name] = grab_data(path, name)

    check_valid_LP(data['A'], data['b'][0], 
                   data['b'][1], data['x'], data['c'][0])

    A, b, Ax_constraints = flip_constraints(data['A'], 
                                            data['b'][0],
                                            data['b'][1])
    data['A'] = A
    data['b'] = b 
    data['Ax_constraints'] = Ax_constraints
    data['problem'] = data['c'][1]
    data['c'] = data['c'][0]

    return(data)
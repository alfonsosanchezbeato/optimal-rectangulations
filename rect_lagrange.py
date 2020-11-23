#!/usr/bin/env python3
import itertools
import numpy as np
import rectangles as rr
import sympy as sp


# Returns value of optimization function
# X: contains [w_1,..,w_N,h_1,..,h_N,l_1,..,l_{N+1}]
#    (widths, heights, and Lagrange multipliers for the N+1 constraints)
# E: restrictions on the rectangulation
# w: background width
# h: background height
# k: desired w_i/h_i aspect ratio
def get_optimization_f_val(X, E, w, h, k):
    n_rect_vars = E.shape[1]
    N = n_rect_vars//2
    n_eqs = E.shape[0]
    indep = np.zeros(n_eqs)
    indep[0] = w
    indep[1] = h
    v = np.matmul(E, X[:n_rect_vars]) - indep
    # Multiply by Lagrange multipliers now
    v = np.dot(v, X[n_rect_vars:])
    # Function to optimize
    for r in range(1, N):
        v += (X[0]*X[N] - X[r]*X[N + r])**2
    # c controls the relation between the two optimization criteria
    c = 0.
    T = c*h*h
    for r in range(N):
        v += T*(X[r] - k*X[N + r])**2
    return v


# Returns the derivatives for the optimization function.
# X: contains [w_1,..,w_N,h_1,..,h_N,l_1,..,l_{N+1}]
#    (widths, heights, and Lagrange multipliers for the N+1 constraints)
# E: restrictions on the rectangulation
# w: background width
# h: background height
# k: desired w_i/h_i aspect ratio
def get_derivative_from_eqs(X, E, w, h, k):
    diff = np.zeros(len(X))
    n_rect_vars = E.shape[1]
    N = n_rect_vars//2
    # c controls the relation between the two optimization criteria.
    c = 0.
    T = c*h*h
    # dF/dw_i
    for i in range(N):
        diff[i] = 2*T*(X[i] - k*X[N + i])
        for j in range(N + 1):
            diff[i] += X[2*N + j]*E[j, i]
    for i in range(1, N):
        diff[i] += -2*X[N + i]*(X[0]*X[N] - X[i]*X[N + i])
    for i in range(1, N):
        diff[0] += 2*X[N]*(X[0]*X[N] - X[i]*X[N + i])
    # dF/dh_i
    for i in range(N):
        diff[N + i] = -2*T*k*(X[i] - k*X[N + i])
        for j in range(N + 1):
            diff[N + i] += X[2*N + j]*E[j, N + i]
    for i in range(1, N):
        diff[N + i] += -2*X[i]*(X[0]*X[N] - X[i]*X[N + i])
    for i in range(1, N):
        diff[N] += 2*X[0]*(X[0]*X[N] - X[i]*X[N + i])
    # dF/dl_j
    n_eqs = E.shape[0]
    indep = np.zeros(n_eqs)
    indep[0] = w
    indep[1] = h
    diff[n_rect_vars:] = np.matmul(E, X[:n_rect_vars]) - indep
    return diff


# Get Nx2 matrix with widths and height from a dictionary returned by
# sympy's solve functions (variables named w<i> and h<i> are expected).
def get_matrix_from_solution(N, sol):
    mat = np.zeros((2, N))
    for sym, val in sol.items():
        if sym.name[0] == 'w':
            mat[0, int(sym.name[1:])] = val
        if sym.name[0] == 'h':
            mat[1, int(sym.name[1:])] = val
    return mat


# Finds the minimal of the optimization function by solving the linear
# system of derivatives analytically, by using sympy. (v2)
def solve_rectangle_eqs(E, w, h, k):
    n_rect_vars = E.shape[1]
    N = n_rect_vars//2
    dF = [0]*(3*N + 1)
    size_avg = w*h/N
    # c controls the relation between the two optimization criteria.
    c = 0.
    T = c*h*h
    # Factor for size
    Q = 1.
    # Add symbols (widths, heights, lambdas)
    W = sp.symbols('w:{}'.format(N), positive=True)
    H = sp.symbols('h:{}'.format(N), positive=True)
    L = sp.symbols('l:{}'.format(N + 1))
    # Create the expressions
    # dF/dw_i
    for i in range(N):
        dF[i] = 2*Q*H[i]*(W[i]*H[i] - size_avg)
        dF[i] += 2*T*(W[i] - k*H[i])
        for j in range(N + 1):
            dF[i] += L[j]*E[j, i]
    # dF/dh_i
    for i in range(N):
        dF[N + i] = 2*Q*W[i]*(W[i]*H[i] - size_avg)
        dF[N + i] += -2*T*k*(W[i] - k*H[i])
        for j in range(N + 1):
            dF[N + i] += L[j]*E[j, N + i]
    # dF/dl_j
    for j in range(N + 1):
        for i in range(N):
            dF[2*N + j] += E[j, i]*W[i] + E[j, N + i]*H[i]
    dF[2*N] += -w
    dF[2*N + 1] += -h
    # Solve the system
    symbols = []
    symbols.extend(W)
    symbols.extend(H)
    symbols.extend(L)
    # 1. nonlinsolve() if Q != 0
    # 2. linsolve() if Q == 0
    # 3. solve() seems to work better than nonlinsolve(), who knows why...
    #    To use it, convert solution to return a FiniteSet.
    # return sp.nonlinsolve(dF, symbols)
    # return sp.linsolve(dF, symbols)
    sol_dict = sp.solve(dF)
    if len(sol_dict) > 1:
        print('Warning: more than one solution')
    mat = get_matrix_from_solution(N, sol_dict[0])
    sol_ls = []
    for i in range(N):
        sol_ls.append(mat[0, i])
    for i in range(N):
        sol_ls.append(mat[1, i])
    return sp.FiniteSet(tuple(sol_ls))


def squared_size_diff(N, sol, mean_sz):
    diff = 0
    for i in range(0, N):
        diff += (sol[0, i]*sol[1, i] - mean_sz)**2
    return diff


# Check if all rectangles in solution are in a given range:
# (1-p)*k < w_i/h_i < k/(1-p)
# k: expected proportiong between width and height
# sol: widths and height in 2xN matrix
# perc: deviation percentage from k, in [0,1) range
def check_proportions_in_range(k, sol, perc):
    c = 1 - perc
    bottom = k*c
    top = k/c
    for i in range(0, sol.shape[1]):
        k_i = sol[0, i]/sol[1, i]
        if k_i <= bottom or k_i >= top:
            return False
    return True


# Compares two rectangulations
# sol_a,sol_b: widths and heights in 2xN matrix form
# w,h: dimensions of the background
# Returns true if A is considered better than B, false otherwise
def compare_rectangulations_size(sol_a, sol_b, w, h):
    N = sol_a.shape[1]
    mean_sz = w*h/N
    diff_a = squared_size_diff(N, sol_a, mean_sz)
    diff_b = squared_size_diff(N, sol_b, mean_sz)
    return diff_a < diff_b


# Calculate best rectangulation in the sense of best aspect ration for N
# rectangles and background size w x h
def get_best_rect_for_window_old(N, k, w, h):
    # Check all permutations
    # TODO filter duplicates
    sol_best = np.zeros(0)
    B_best = None
    dev_from_k = 0.15
    dev_from_k = 0.5
    seq_first = [r for r in range(0, N)]
    # XXX
    # seq_first = [1, 4, 2, 0, 3]
    for seq in itertools.permutations(seq_first):
        B, dim = rr.do_diagonal_rectangulation(seq)
        E = rr.build_rectangulation_equations(B)
        print(seq)
        print(B)
        sol_sympy = solve_rectangle_eqs(E, w, h, k)
        sol = np.zeros((2, N))
        # Only one solution
        print(sol_sympy)
        full_sol = True
        for i in range(N):
            if not isinstance(sol_sympy.args[0][i], sp.Float) or \
               not isinstance(sol_sympy.args[0][N + i], sp.Float):
                print('Not fully determined:', sol_sympy)
                full_sol = False
                break
            sol[0, i] = sol_sympy.args[0][i]
            sol[1, i] = sol_sympy.args[0][N + i]
        # sol = get_matrix_from_solution(N, sol_sympy)
        if full_sol and check_proportions_in_range(k, sol, dev_from_k):
            if sol_best.size == 0 or \
               compare_rectangulations_size(sol, sol_best, w, h):
                sol_best = sol
                B_best = B
        rr.draw_resized_rectangles(B, sol, w, h)

    return B_best, sol_best

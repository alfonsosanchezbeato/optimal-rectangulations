#!/usr/bin/env python3
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import sympy as sp


class RectangleLayout:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def max_x(self):
        return self.x + self.width

    def max_y(self):
        return self.y + self.height

    def __eq__(self, other):
        return (self.x, self.y, self.width, self.height) \
            == (other.x, other.y, other.width, other.height)

    def __str__(self):
        return 'Origin: (' + str(self.x) + ',' + str(self.y) + '). W: ' \
            + str(self.width) + '. H: ' + str(self.height)


# We consider a cell taken if its value points to a rectangle or if
# outside background.
def is_b_empty(B, N, i, j):
    if i < 0 or i >= N or j < 0 or j >= N:
        return False
    return B[i, j] < 0


# Do a diagonal rectangulation
# seq: sequence with order in which to create the rectangles.
#      Lenght N, with values between 0 and N-1.
def do_diagonal_rectangulation(seq):
    N = len(seq)
    # Background rectangle, as a grid where we will indicate the ovelaying
    # rectangle in each cell (each of them can span acrosss multiple cells).
    B = np.full((N, N), -1, dtype=int)
    for r in seq:
        # Top-left corner (top, left)
        top = r
        left = r
        if is_b_empty(B, N, r-1, r-1):
            # Go left
            while is_b_empty(B, N, top, left - 1):
                left -= 1
        else:
            # Go up
            while top - 1 >= 0 and not is_b_empty(B, N, top - 1, left - 1):
                top -= 1
        # Bottom-right corner
        bottom = r
        right = r
        if is_b_empty(B, N, r+1, r+1):
            # Go down
            while is_b_empty(B, N, bottom + 1, right):
                bottom += 1
        else:
            # Go right
            while right + 1 <= N - 1 and \
                  not is_b_empty(B, N, bottom + 1, right + 1):
                right += 1
        # Fill inside B
        # print('Rect: ({},{}), ({},{})'.format(top, left, bottom, right))
        for i in range(top, bottom + 1):
            for j in range(left, right + 1):
                B[i, j] = r

    return B


# Creates equations for the rectangulation restrictions
# B: rectangulation as created by do_diagonal_rectangulation
# w: full width
# h: full height
def build_rectangulation_equations(B):
    N = B.shape[0]
    # Top and left of rectangulation
    E = np.zeros((2, 2*N))
    for j in range(N):
        E[0, B[0, j]] = 1
    for i in range(N):
        E[1, N + B[i, 0]] = 1
    # Horizontal segments
    eq = None
    for i in range(N - 1):
        for j in range(N):
            if B[i, j] != B[i + 1, j]:
                # Different rectangle up and down
                if eq is None:
                    eq = np.zeros(2*N)
                eq[B[i, j]] = 1
                eq[B[i + 1, j]] = -1
            else:
                # Close equation if we were creating one
                if eq is not None:
                    E = np.vstack([E, eq])
                    eq = None
        if eq is not None:
            E = np.vstack([E, eq])
            eq = None
    # Vertical segments
    for j in range(N - 1):
        for i in range(N):
            if B[i, j] != B[i, j + 1]:
                # Different rectangle left and right
                if eq is None:
                    eq = np.zeros(2*N)
                eq[N + B[i, j]] = 1
                eq[N + B[i, j + 1]] = -1
            else:
                # Close equation if we were creating one
                if eq is not None:
                    E = np.vstack([E, eq])
                    eq = None
        if eq is not None:
            E = np.vstack([E, eq])
            eq = None
    return E


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
    # T controls the relation between the two optimization criteria
    T = 0.
    for r in range(N):
        v += T*(X[r] - k*X[N + r])**2
    return v

# Derivative by differences, not using it now
# def get_derivative(E, X, w, h, k):
#     dLambda = np.zeros(len(X))
#     # Step size
#     step = 1e-3
#     for i in range(len(X)):
#         dX = np.zeros(len(X))
#         dX[i] = step
#         dLambda[i] = (get_optimization_f_val(X + dX, E, w, h, k)
#                       - get_optimization_f_val(X - dX, E, w, h, k))/(2*step)
#     return dLambda


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
    # T controls the relation between the two optimization criteria.
    # It needs to be quite big to be dominant (~100000)
    # T = 100000.
    T = 0.
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


# Finds the minimal of the optimization function by solving the linear
# system of derivatives analytically, by using sympy. (v2)
def solve_rectangle_eqs(E, w, h, k):
    n_rect_vars = E.shape[1]
    N = n_rect_vars//2
    dF = [0]*(3*N + 1)
    size_avg = w*h/N
    # T controls the relation between the two optimization criteria.
    # It needs to be quite big to be dominant (~100000)
    T = 1.
    # Factor for size
    Q = 0.
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
    return sp.nonlinsolve(dF, symbols)
    # return sp.linsolve(dF, symbols)
    # sol_dict = sp.solve(dF)
    # if len(sol_dict) > 1:
    #     print('Warning: more than one solution')
    # mat = get_matrix_from_solution(N, sol_dict[0])
    # sol_ls = []
    # for i in range(N):
    #     sol_ls.append(mat[0, i])
    # for i in range(N):
    #     sol_ls.append(mat[1, i])
    # return sp.FiniteSet(tuple(sol_ls))


def _len_rect_lines(rect_lines):
    for l in rect_lines:
        return len(l)
    return 0


# Solves by fixing either width or height of each rectangle to the same
# values across the rectangulation.
def solve_fit_rectangles(E, B, w, h, k):
    n_rect_vars = E.shape[1]
    N = n_rect_vars//2
    # Append column with independent term
    indep = np.zeros((E.shape[0], 1))
    indep[0] = w
    indep[1] = h
    E = np.hstack([E, indep])

    # Fix horizontally
    max_rect_lines = []
    for i in range(N):
        line_r = []
        for j in range(N):
            if B[i, j] in line_r:
                continue
            line_r.append(B[i, j])
        num_curr_lines = _len_rect_lines(max_rect_lines)
        if len(line_r) > num_curr_lines:
            max_rect_lines = []
            max_rect_lines.append(line_r)
        elif len(line_r) == num_curr_lines:
            # TODO check not same elements
            max_rect_lines.append(line_r)
    # w_min = w/len(max_rect_lines[0])
    # Add equations
    # w_i = w_k = w_min?
    for l in max_rect_lines:
        for i, r in enumerate(l[:-1]):
            eq = np.zeros(2*N + 1)
            eq[r] = 1
            eq[l[i + 1]] = -1
            E = np.vstack([E, eq])

    # Fix vertically
    max_rect_lines = []
    for j in range(N):
        line_r = []
        for i in range(N):
            if B[i, j] in line_r:
                continue
            line_r.append(B[i, j])
        num_curr_lines = _len_rect_lines(max_rect_lines)
        if len(line_r) > num_curr_lines:
            max_rect_lines = []
            max_rect_lines.append(line_r)
        elif len(line_r) == num_curr_lines:
            max_rect_lines.append(line_r)
    # h_min = h/len(max_rect_lines[0])
    # Add equations
    for l in max_rect_lines:
        for i, r in enumerate(l[:-1]):
            eq = np.zeros(2*N + 1)
            eq[N + r] = 1
            eq[N + l[i + 1]] = -1
            E = np.vstack([E, eq])

    # return np.linalg.solve(E[:, :2*N], E[:, 2*N:])
    return sp.linsolve(sp.Matrix(E))


# Draw rectangulation
# B: rectangulation square
# w_pix: width in pixels
# h_pix: height in pixels
def draw_rectangles(B, w_pix, h_pix):
    # Go horizontal line by line
    Nx = B.shape[0]
    Ny = B.shape[1]
    w_step = w_pix/Nx
    h_step = h_pix/Ny
    plt.figure()
    xs = [0, w_pix]
    ys = [h_pix, h_pix]
    plt.plot(xs, ys)
    xs = [0, 0]
    ys = [h_pix, 0]
    plt.plot(xs, ys)
    xs = [w_pix, w_pix]
    ys = [0, h_pix]
    plt.plot(xs, ys)
    xs = [0, w_pix]
    ys = [0, 0]
    plt.plot(xs, ys)
    for i in range(Ny):
        for j in range(Nx):
            if j > 0 and B[i, j] != B[i, j - 1]:
                # Vertical line
                xs = [j*w_step, j*w_step]
                ys = [h_pix - i*h_step, h_pix - (i + 1)*h_step]
                plt.plot(xs, ys)
            if i > 0 and B[i, j] != B[i - 1, j]:
                # Horizontal line
                xs = [j*w_step, (j + 1)*w_step]
                ys = [h_pix - i*h_step, h_pix - i*h_step]
                plt.plot(xs, ys)
    # Bottom border
    for j in range(1, Nx):
        if B[Ny - 1, j] != B[Ny - 1, j - 1]:
            # Vertical line
            xs = [j*w_step, j*w_step]
            ys = [h_step, 0]
            plt.plot(xs, ys)
    # Right border
    for i in range(1, Ny):
        if B[i, Nx - 1] != B[i - 1, Nx - 1]:
            # Horizontal line
            xs = [(Nx - 1)*w_step, Nx*w_step]
            ys = [h_pix - i*h_step, h_pix - i*h_step]
            plt.plot(xs, ys)

    plt.show()


# Draw rectangulation. We get the relative positions of the rectangles
# from B, and the real dimension from D. Altough relative positions of
# rectangles might have changed with the new sizes, the relative
# positions of rectangles across lines defined by B do not change. We
# use that to draw the rectangles.
# B: rectangulation square
# D: Vector with dimensions (w, h) in each cell
# w_pix: width in pixels
# h_pix: height in pixels
def draw_resized_rectangles(B, D, w_pix, h_pix):
    Nx = B.shape[0]
    Ny = B.shape[1]
    fig, ax = plt.subplots()
    Rs = {}

    # Horizontal segment by horizontal segment (we could use the
    # vertical ones as well).
    for i in range(Ny):
        for j in range(Nx):
            # 1. We are right below an horizontal line
            # 2. We have just moved to a new rectangle
            # -> we draw the rectangle
            if (i == 0 or B[i, j] != B[i - 1, j]) and \
               (j == 0 or B[i, j] != B[i, j - 1]):
                if j == 0:
                    r_x = 0
                else:
                    # Use left rectangle
                    r_x = Rs[B[i, j - 1]].max_x()
                if i == 0:
                    r_y = 0
                else:
                    # Use top rectangle
                    r_y = Rs[B[i - 1, j]].max_y()
                r_w = D[0, B[i, j]]
                r_h = D[1, B[i, j]]
                Rs[B[i, j]] = RectangleLayout(r_x, r_y, r_w, r_h)
                rect = patches.Rectangle((r_x, r_y), r_w, r_h,
                                         facecolor='w', edgecolor='k')
                ax.add_patch(rect)

    ax.set_xlim(0, w_pix)
    ax.set_ylim(h_pix, 0)
    ax.set_aspect('equal')
    plt.show()


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
def get_best_rect_for_window(N, k, w, h):
    # Check all permutations
    # TODO filter duplicates
    sol_best = np.zeros(0)
    B_best = None
    dev_from_k = 0.15
    seq_first = [r for r in range(0, N)]
    for seq in itertools.permutations(seq_first):
        B = do_diagonal_rectangulation(seq)
        E = build_rectangulation_equations(B)
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
        draw_resized_rectangles(B, sol, w, h)

    return B_best, sol_best


# Data modelling.
# N: final number of squares
# Equations contain 2*N variables: w1,..,wN,h1,..,hN
# Final number of equations will be N+1

if __name__ == '__main__':
    N = 3
    k = 1.5
    w = 400
    h = 200
    B, sol = get_best_rect_for_window(N, k, w, h)
    if sol.size:
        print("Best solution is", sol)
        draw_resized_rectangles(B, sol, w, h)
    else:
        print("No solution found")

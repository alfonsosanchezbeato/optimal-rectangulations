#!/usr/bin/env python3
import numpy as np
import sympy as sp
from fractions import Fraction


# class RectLimitSegment:
#     def __init__(self, rect, ):


class RectangleLayout:
    def __init__(self, x, y, width, height):
        # Fractions that define relative positions in the background
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        # Lists with nearby rectangles
        self.top = []
        self.right = []
        self.bottom = []
        self.left = []

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

    # Is the side "closed"? (no space for more rectangles)

    def top_closed(self):
        if self.top and self.top[-1].max_x() >= self.max_x():
            return True
        return False

    def right_closed(self):
        if self.right and self.top[-1].max_y() >= self.max_y():
            return True
        return False

    def bottom_closed(self):
        if self.bottom and self.bottom[-1].max_x() >= self.max_x():
            return True
        return False

    def left_closed(self):
        if self.left and self.left[-1].max_y() >= self.max_y():
            return True
        return False


def get_rects_corner(R):
    x = Fraction(-1, 1)
    y = Fraction(-1, 1)
    for r in R:
        # Check first if space left to the right, then if
        # 1. No space, go to next (if corner we will find out later
        #    when looking at rectangles at bottom)
        # 2. Already a right rectangle, but some space: we
        #    know then the two rectangles that form the corner
        # 3. Empty: go to top, then if
        #    1. Top not closed, no corner, go to next
        #    2. Top closed, look at last rectangle there
        #       1. Last rectangle expands further to the right: corner
        #       2. Otherwise, look at its right. If closed, corner.
        if r.right_closed():
            continue
        if r.right:
            x = r.max_x()
            y = r.right[-1].max_y()
            break
        # Empty right, check top
        if not r.top_closed():
            continue
        top_r = r.top[-1]
        if top_r.max_x() > r.max_x():
            x = r.max_x()
            y = r.y
            break
        if top_r.right_closed():
            x = r.max_x()
            y = r.y
            break

    return x, y


def get_topleft_corner(B, R):
    x = Fraction(-1, 1)
    y = Fraction(-1, 1)
    # Top of B
    if not B.top_closed():
        if not B.top:
            x = Fraction(0, 1)
        else:
            x = B.top[-1].max_x()
        y = Fraction(0, 1)
    elif not B.left_closed():
        # Left of B
        x = Fraction(0, 1)
        if not B.left:
            y = Fraction(0, 1)
        else:
            y = B.left[-1].max_y()
    else:
        # Corners limited by regular rectangles
        x, y = get_rects_corner(R)

    return x, y


def _split_rectangle():
    # Background rectangle
    B = RectangleLayout(Fraction(0, 1), Fraction(0, 1),
                        Fraction(1, 1), Fraction(1, 1))
    # Rectangles list
    R = []
    x, y = get_topleft_corner(B, R)
    print("Next corner:", x, ",", y)


# r_idx: index to rectangle to be split
# N: final number of rectangles
# R: current list of rectangles, a new one will be appended
# E: list of equations, it will be modified
def split_rectangle_vertically(r_idx, N, R, E):
    # Modify r and create a new rectangle
    r = R[r_idx]
    r.width /= 2
    r2 = RectangleLayout(r.x + r.width, r.y, r.width, r.height)
    index = len(R)
    R.append(r2)
    # Modify equations appropriately
    # w_old -> w_1 + w_2
    for eq in E:
        if eq[r_idx] != 0:
            eq[index] = eq[r_idx]
    # h_old -> h_2 iff line to the right of r_old
    for eq in E:
        v_1 = eq[N + r_idx]
        if v_1 != 0:
            # Different sign implies rectange on different side to r
            for i, v in enumerate(eq[N:]):
                # We only have 0, 1 or -1 as possible values
                if v != 0 and v != v_1 and R[r_idx].x < R[i].x:
                    eq[N + index] = v_1
                    eq[N + r_idx] = 0
    # h_1 - h_2 = 0
    new_eq = np.zeros(2*N)
    new_eq[N + r_idx] = 1
    new_eq[N + index] = -1
    E = np.vstack([E, new_eq])
    return R, E


# r_idx: index to rectangle to be split
# N: final number of rectangles
# R: current list of rectangles, a new one will be appended
# E: list of equations, it will be modified
def split_rectangle_horizontally(r_idx, N, R, E):
    # Modify r and create a new rectangle
    r = R[r_idx]
    r.height /= 2
    r2 = RectangleLayout(r.x, r.y + r.height, r.width, r.height)
    index = len(R)
    R.append(r2)
    # Modify equations appropriately
    # h_old -> h_1 + h_2
    for eq in E:
        if eq[N + r_idx] != 0:
            eq[N + index] = eq[N + r_idx]
    # w_old -> w_2 iff line below r_old
    for eq in E:
        v_1 = eq[r_idx]
        if v_1 != 0:
            # Different sign implies rectange on different side to r
            for i, v in enumerate(eq[:N]):
                # We only have 0, 1 or -1 as possible values
                if v != 0 and v != v_1 and R[r_idx].y < R[i].y:
                    eq[index] = v_1
                    eq[r_idx] = 0
    # w_1 - w_2 = 0
    new_eq = np.zeros(2*N)
    new_eq[r_idx] = 1
    new_eq[index] = -1
    E = np.vstack([E, new_eq])
    return R, E


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


# E: restrictions on the rectangulation
# X: contains [w_1,..,w_N,h_1,..,h_N,l_1,..,l_{N+1}]
#    (widths, heights, and Lagrange multipliers for the N+1 constraints)
# w: background width
# h: background height
# k: desired w_i/h_i aspect ratio
def get_optimal_rectangles(E, X, w, h, k):
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
    T = 0.5
    for r in range(N):
        v += T*(X[r] - k*X[N + r])**2
    return v


# Returns the derivatives for the optimization function.
# E: restrictions on the rectangulation
# X: contains [w_1,..,w_N,h_1,..,h_N,l_1,..,l_{N+1}]
#    (widths, heights, and Lagrange multipliers for the N+1 constraints)
# w: background width
# h: background height
# k: desired w_i/h_i aspect ratio
def get_derivative_from_eqs(E, X, w, h, k):
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
    print(dF)
    return sp.solve(dF)


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

    print(E)
    # return np.linalg.solve(E[:, :2*N], E[:, 2*N:])
    return sp.linsolve(sp.Matrix(E))


def get_derivative(E, X, w, h, k):
    dLambda = np.zeros(len(X))
    # Step size
    step = 1e-3
    for i in range(len(X)):
        dX = np.zeros(len(X))
        dX[i] = step
        dLambda[i] = (get_optimal_rectangles(E, X + dX, w, h, k)
                      - get_optimal_rectangles(E, X - dX, w, h, k))/(2*step)
    return dLambda


# Data modelling.
# N: final number of squares
# Equations contain 2*N variables: w1,..,wN,h1,..,hN
# Final number of equations will be N+1

if __name__ == '__main__':
    R = [RectangleLayout(Fraction(0, 1), Fraction(0, 1),
                         Fraction(1, 1), Fraction(1, 1))]
    _split_rectangle()

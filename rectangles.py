#!/usr/bin/env python3
import itertools
import math
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import scipy.optimize as opt


# Helper class with rectangle information, used for the moment only when
# drawing rectangulations.
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
#      Length N, with values between 0 and N-1.
# Returns a numpy matrix where each cell contains an integer for the
# rectangle in that position and a 2xN matrix with the dimensions of
# each rectangle as a percentage of the bounding segments.
def do_diagonal_rectangulation(seq):
    N = len(seq)
    # Background rectangle, as a grid where we will indicate the overlaying
    # rectangle in each cell (each of them can span acrosss multiple cells).
    B = np.full((N, N), -1, dtype=int)
    dim = np.zeros((2, N))

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

        dim[0, r] = (right - left + 1)/N
        dim[1, r] = (bottom - top + 1)/N
        # Fill inside B
        for i in range(top, bottom + 1):
            for j in range(left, right + 1):
                B[i, j] = r

    return B, dim


# Creates equations for the rectangulation restrictions, without the
# independent terms.
# B: rectangulation as created by do_diagonal_rectangulation
def build_rectangulation_equations(B):
    N = B.shape[0]
    E = np.zeros((2, 2*N))
    # Top and left of rectangulation
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


def opt_f_val(X, w, h, k, c):
    N = len(X)//2
    v = 0
    avg_sz = w*h/N
    T = c*h*h
    # Function to optimize
    for r in range(N):
        v += (X[r]*X[N + r] - avg_sz)**2
    for r in range(N):
        v += T*(X[r] - k*X[N + r])**2
    return v


def opt_jac_val(X, w, h, k, c):
    N = len(X)//2
    diff = np.zeros(2*N)
    avg_sz = w*h/N
    T = c*h*h
    # dF/dw_i
    for i in range(N):
        diff[i] = 2*T*(X[i] - k*X[N + i])
        diff[i] += 2*X[N + i]*(X[i]*X[N + i] - avg_sz)
    # dF/dh_i
    for i in range(N):
        diff[N + i] = -2*T*k*(X[i] - k*X[N + i])
        diff[N + i] += 2*X[i]*(X[i]*X[N + i] - avg_sz)

    return diff


# Variables to optimize are [w_1,..,w_N,h_1,..,h_N]
# E: rectangulation equations coefficients (N+1)x2N matrix
# est: initial estimation for the sizes in a 2xN matrix
# w: width of bounding rectangle
# h: height of bounding rectangle
# k: desired w_i/h_i ratio
# c: controls the balance between the two optimization criteria.
def minimize_rectangulation(E, est, w, h, k, c):
    n_rect_vars = E.shape[1]
    N = n_rect_vars//2
    initial_est = est.reshape(est.size)

    n_cons_eqs = E.shape[0]
    indep = np.zeros(n_cons_eqs)
    indep[0] = w
    indep[1] = h
    constr = opt.LinearConstraint(E, indep, indep)

    sol = opt.minimize(opt_f_val, initial_est, args=(w, h, k, c),
                       jac=opt_jac_val, constraints=constr)
    if not sol.success:
        print('Could not optimize:', sol.message)
    ret = np.zeros((2, N))
    ret[0, :] = sol.x[:N]
    ret[1, :] = sol.x[N:]
    return ret


# Draw rectangulation. We get the relative positions of the rectangles
# from B, and the real dimension from D. Altough relative positions of
# rectangles might have changed with the new sizes, the relative
# positions of rectangles across segments defined by B do not change. We
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


# Calculate best rectangulation in the sense of equal area distribution and
# best aspect ration for N rectangles and background size w x h.
def get_best_rect_for_window(N, k, w, h):
    # Check all permutations
    # TODO filter duplicates
    f_best = sys.float_info.max
    sol_best = np.zeros(0)
    B_best = None
    seq_best = None
    # 0.1: proportion is predominant
    # 0.05: seems well-balanced
    c = 0.05
    # If the new optimum is only very slightly better, keep the old
    # value so solutions are more congruent when the parameters change
    # slightly (itertools.permutations() always produces sequences in
    # the same order). This delta is the allowed difference.
    delta = (w*h)**2/1e12
    seq_first = [r for r in range(0, N)]
    for seq in itertools.permutations(seq_first):
        if not is_baxter_permutation(seq):
            continue

        B, dim = do_diagonal_rectangulation(seq)
        E = build_rectangulation_equations(B)
        dim[0, :] *= w
        dim[1, :] *= h
        sol = minimize_rectangulation(E, dim, w, h, k, c)

        vals = sol[0, :].tolist()
        vals.extend(sol[1, :].tolist())
        f_val = opt_f_val(vals, w, h, k, c)
        if f_val + delta < f_best:
            f_best = f_val
            sol_best = sol
            B_best = B
            seq_best = seq

    return B_best, sol_best, seq_best


# Returns True if subseq matches the pattern
# pattern: contains numbers
# gaps: booleans list, gaps[i]=true if dash before pattern[i]
def subseq_matches_pattern(seq, subseq, pattern, gaps):
    # i is seq index, j is subseq/pattern index
    j = 0
    gap = False
    len_subseq = len(subseq)
    for i, sv in enumerate(seq):
        if sv != subseq[j]:
            gap = True
            continue

        # No match if gap found and no dash
        if j > 0 and gaps[j] is False and gap is True:
            return False

        # Check previous values in subseq
        for k in range(0, j - 1):
            if pattern[k] < pattern[j] and subseq[k] >= subseq[j]:
                return False
            if subseq[k] < subseq[j] and pattern[k] >= pattern[j]:
                return False

        gap = False
        j += 1
        if j == len_subseq:
            break

    return True


def get_subsequence(n, seq):
    if n == 0:
        yield []
    for i, s in enumerate(seq):
        for l in get_subsequence(n - 1, seq[i + 1:]):
            subseq = [s]
            subseq.extend(l)
            yield subseq


def is_baxter_permutation(seq):
    is_baxter = True
    for subseq in get_subsequence(4, seq):
        if subseq_matches_pattern(seq, subseq, [3, 1, 4, 2],
                                  [False, True, False, True]) or \
           subseq_matches_pattern(seq, subseq, [2, 4, 1, 3],
                                  [False, True, False, True]):
            is_baxter = False
            break

    return is_baxter


def count_number_diagonal_rects(N):
    num_rects = 0
    seq_first = [r for r in range(0, N)]
    for seq in itertools.permutations(seq_first):
        if is_baxter_permutation(seq):
            num_rects += 1

    return num_rects


def get_best_for_N():
    # Try 3, 7...
    N = 7
    k = 1.5
    w = 400
    h = 200
    B, sol, seq = get_best_rect_for_window(N, k, w, h)
    if sol.size:
        print("Best solution is", sol)
        draw_resized_rectangles(B, sol, w, h)
    else:
        print("No solution found")


def plot_for_N5():
    num_pt = 15
    # For instance 320x180
    num_pix = 57600
    aspect_lb = 0.3
    aspect_ub = 4.
    N = 5
    # Usual camera x/y ratio
    k = 1.33
    for aspect in np.linspace(aspect_lb, aspect_ub, num_pt):
        w = math.sqrt(aspect*num_pix)
        h = w/aspect
        B, sol, seq = get_best_rect_for_window(N, k, w, h)
        print(seq)
        draw_resized_rectangles(B, sol, w, h)


if __name__ == '__main__':
    for ss in get_subsequence(3, [1, 2, 3, 4]):
        print(ss)
    print(is_baxter_permutation([1, 2, 3, 4]))
    print(is_baxter_permutation([3, 1, 4, 2]))
    print(is_baxter_permutation([2, 4, 1, 3]))
    for N in range(1, 8):
        print(count_number_diagonal_rects(N))
    plot_for_N5()

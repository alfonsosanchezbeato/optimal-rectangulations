#!/usr/bin/env python3
import numpy as np
import sympy as sp


def _len_rect_lines(rect_lines):
    for l in rect_lines:
        return len(l)
    return 0


# Solves by fixing either width or height of each rectangle to the same
# values across the rectangulation. Returns a FiniteSet.
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

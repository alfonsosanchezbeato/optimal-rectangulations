#!/usr/bin/env python3
import numpy as np


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

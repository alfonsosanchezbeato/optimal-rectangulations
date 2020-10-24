import numpy as np
import unittest
import rectangles as r
from scipy.optimize import fsolve


class TestStringMethods(unittest.TestCase):

    def print_rects(self, R):
        print('\nRectangles:')
        for ri in R:
            print(ri)

    def test_diagonal_rectangulation_3rect(self):
        B = r.do_diagonal_rectangulation([0, 1, 2])
        print(B)
        Bc = np.array([[0, 1, 2],
                       [0, 1, 2],
                       [0, 1, 2]],
                      dtype=int)
        self.assertTrue((B == Bc).all())
        E = r.build_rectangulation_equations(B)
        print(E)
        Ec = np.array([[1.,  1.,  1.,  0.,  0.,  0.],
                       [0.,  0.,  0.,  1.,  0.,  0.],
                       [0.,  0.,  0.,  1., -1.,  0.],
                       [0.,  0.,  0.,  0.,  1., -1.]])
        self.assertTrue((E == Ec).all())
        w = 400
        h = 200
        k = 1.5
        initial_est = np.ones(3*2 + 4)
        for v in range(3):
            initial_est[v] = w/3
        for v in range(3, 6):
            initial_est[v] = h/3

        def dfunc(X): return r.get_derivative_from_eqs(E, X, w, h, k)

        X = fsolve(dfunc, initial_est)
        print(X, r.get_optimal_rectangles(E, X, w, h, k))
        print("Using sympy:")
        print(r.solve_rectangle_eqs(E, w, h, k))
        print("Fitting rectangles:")
        print(r.solve_fit_rectangles(E, B, w, h, k))

        B = r.do_diagonal_rectangulation([2, 1, 0])
        print(B)
        Bc = np.array([[0, 0, 0],
                       [1, 1, 1],
                       [2, 2, 2]],
                      dtype=int)
        self.assertTrue((B == Bc).all())
        E = r.build_rectangulation_equations(B)
        print(E)
        X = fsolve(dfunc, initial_est)
        print(X, r.get_optimal_rectangles(E, X, w, h, k))
        # print("Using sympy:")
        # print(r.solve_rectangle_eqs(E, w, h, k))
        print("Fitting rectangles:")
        print(r.solve_fit_rectangles(E, B, w, h, k))

        B = r.do_diagonal_rectangulation([1, 0, 2])
        print(B)
        Bc = np.array([[0, 0, 2],
                       [1, 1, 2],
                       [1, 1, 2]],
                      dtype=int)
        self.assertTrue((B == Bc).all())
        E = r.build_rectangulation_equations(B)
        print(E)
        # With this method we need quite good initial values...
        initial_est[0] = 330
        initial_est[1] = 350
        initial_est[2] = 166
        initial_est[3] = 100
        initial_est[4] = 120
        initial_est[5] = 66
        initial_est[0] = 2*400/3
        initial_est[1] = 2*400/3
        initial_est[2] = 400/3
        initial_est[3] = 200/3
        initial_est[4] = 2*200/3
        initial_est[5] = 200
        X = fsolve(dfunc, initial_est)
        print(X, r.get_optimal_rectangles(E, X, w, h, k))
        # print("Using sympy:")
        # print(r.solve_rectangle_eqs(E, w, h, k))
        print("Fitting rectangles:")
        print(r.solve_fit_rectangles(E, B, w, h, k))

    def test_diagonal_rectangulation_5rect(self):
        B = r.do_diagonal_rectangulation([2, 0, 4, 1, 3])
        print(B)
        Bc = np.array([[0, 1, 1, 3, 3],
                       [0, 1, 1, 3, 3],
                       [2, 2, 2, 3, 3],
                       [2, 2, 2, 3, 3],
                       [2, 2, 2, 4, 4]],
                      dtype=int)
        self.assertTrue((B == Bc).all())
        E = r.build_rectangulation_equations(B)
        print(E)
        w = 400
        h = 200
        k = 1.5
        # print("Using sympy:")
        # print(r.solve_rectangle_eqs(E, w, h, k))
        print("Fitting rectangles:")
        print(r.solve_fit_rectangles(E, B, w, h, k))

    def _test_diagonal_rectangulation_15rect(self):
        B = r.do_diagonal_rectangulation(
            [7, 12, 6, 4, 10, 1, 13, 5, 14, 8, 9, 2, 0, 3, 11])
        Bc = np.array([[0, 0, 0, 3, 3, 3, 3, 3,  3,  3,  3, 11, 11, 11, 11],
                       [1, 1, 2, 3, 3, 3, 3, 3,  3,  3,  3, 11, 11, 11, 11],
                       [1, 1, 2, 3, 3, 3, 3, 3,  3,  3,  3, 11, 11, 11, 11],
                       [1, 1, 2, 3, 3, 3, 3, 3,  3,  3,  3, 11, 11, 11, 11],
                       [4, 4, 4, 4, 4, 5, 5, 5,  8,  9,  9, 11, 11, 11, 11],
                       [4, 4, 4, 4, 4, 5, 5, 5,  8,  9,  9, 11, 11, 11, 11],
                       [6, 6, 6, 6, 6, 6, 6, 6,  8,  9,  9, 11, 11, 11, 11],
                       [7, 7, 7, 7, 7, 7, 7, 7,  8,  9,  9, 11, 11, 11, 11],
                       [7, 7, 7, 7, 7, 7, 7, 7,  8,  9,  9, 11, 11, 11, 11],
                       [7, 7, 7, 7, 7, 7, 7, 7,  8,  9,  9, 11, 11, 11, 11],
                       [7, 7, 7, 7, 7, 7, 7, 7, 10, 10, 10, 11, 11, 11, 11],
                       [7, 7, 7, 7, 7, 7, 7, 7, 10, 10, 10, 11, 11, 11, 11],
                       [7, 7, 7, 7, 7, 7, 7, 7, 12, 12, 12, 12, 12, 13, 14],
                       [7, 7, 7, 7, 7, 7, 7, 7, 12, 12, 12, 12, 12, 13, 14],
                       [7, 7, 7, 7, 7, 7, 7, 7, 12, 12, 12, 12, 12, 13, 14]],
                      dtype=int)
        self.assertTrue((B == Bc).all())
        E = r.build_rectangulation_equations(B)
        print(E)


if __name__ == '__main__':
    unittest.main()

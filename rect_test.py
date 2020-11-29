#!/usr/bin/env python3
import math
import numpy as np
import unittest
import rectangles as r
import rect_lagrange as rl
import rect_fit_sides as rfs
from random import shuffle
from scipy.optimize import fsolve


class TestRectangles(unittest.TestCase):

    def print_rects(self, R):
        print('\nRectangles:')
        for ri in R:
            print(ri)

    def get_matrix_from_finiteset(self, N, sol_set):
        mat = np.zeros((2, N))
        if len(sol_set) != 1:
            print('Warning: more than one solution')
        for sol in sol_set:
            break
        for i in range(N):
            mat[0, i] = sol[i]
        for i in range(N):
            mat[1, i] = sol[N + i]
        return mat

    # Solve rectangulations numerically with Lagrange multipliers.
    # With this method we need quite good initial values...
    def test_lagrange_method_numerical(self):
        w = 320
        h = 180
        k = 1.5
        diagonals = [[0, 1, 2], [2, 1, 0], [1, 0, 2]]
        for diag in diagonals:
            N = len(diag)
            B, dim = r.do_diagonal_rectangulation(diag)
            dim[0, :] *= w
            dim[1, :] *= h
            E = r.build_rectangulation_equations(B)
            # Number of lambda values will be same as number of eqs (N+1)
            initial_est = np.ones(dim.size + N + 1)
            initial_est[0:dim.size] = dim.reshape(dim.size)

            def dfunc(X): return rl.get_derivative_from_eqs(X, E, w, h, k)

            X = fsolve(dfunc, initial_est)
            self.assertAlmostEqual(0.,
                                   rl.get_optimization_f_val(X, E, w, h, k))

    # Solve rectangulations analitycally (using sympy) with Lagrange
    # multipliers.
    def test_lagrange_method_analitycal(self):
        w = 320
        h = 180
        k = 1.5
        # [2, 0, 4, 1, 3] can be handled, but takes a little bit
        diagonals = [[0, 1, 2], [2, 1, 0], [1, 0, 2]]
        for diag in diagonals:
            N = len(diag)
            B, dim = r.do_diagonal_rectangulation(diag)
            dim[0, :] *= w
            dim[1, :] *= h
            E = r.build_rectangulation_equations(B)

            sol = rl.solve_rectangle_eqs(E, w, h, k)
            print(sol)
            mat_sol = self.get_matrix_from_finiteset(N, sol)
            r.draw_resized_rectangles(B, mat_sol, w, h)

    def test_fit_sides(self):
        w = 320
        h = 180
        k = 1.5
        diagonals = [[0, 1, 2], [2, 1, 0], [1, 0, 2], [2, 0, 4, 1, 3]]
        for diag in diagonals:
            N = len(diag)
            B, dim = r.do_diagonal_rectangulation(diag)
            dim[0, :] *= w
            dim[1, :] *= h
            E = r.build_rectangulation_equations(B)
            sol = rfs.solve_fit_rectangles(E, B, w, h, k)
            mat_sol = self.get_matrix_from_finiteset(N, sol)
            r.draw_resized_rectangles(B, mat_sol, w, h)

    def test_scipy_minimize(self):
        w = 320
        h = 180
        k = 1.5
        c = 0.05

        Bc = [np.array([[0, 1, 2],
                        [0, 1, 2],
                        [0, 1, 2]], dtype=int),
              np.array([[0, 0, 0],
                        [1, 1, 1],
                        [2, 2, 2]], dtype=int),
              np.array([[0, 0, 2],
                        [1, 1, 2],
                        [1, 1, 2]], dtype=int),
              np.array([[0, 1, 1, 3, 3],
                        [0, 1, 1, 3, 3],
                        [2, 2, 2, 3, 3],
                        [2, 2, 2, 3, 3],
                        [2, 2, 2, 4, 4]], dtype=int)]
        Ec = [[[1.,  1.,  1.,  0.,  0.,  0.],
               [0.,  0.,  0.,  1.,  0.,  0.],
               [0.,  0.,  0.,  1., -1.,  0.],
               [0.,  0.,  0.,  0.,  1., -1.]],
              [[1.,  0.,  0.,  0.,  0.,  0.],
               [0.,  0.,  0.,  1.,  1.,  1.],
               [1., -1.,  0.,  0.,  0.,  0.],
               [0.,  1., -1.,  0.,  0.,  0.]],
              [[1.,  0.,  1.,  0.,  0.,  0.],
               [0.,  0.,  0.,  1.,  1.,  0.],
               [1., -1.,  0.,  0.,  0.,  0.],
               [0.,  0.,  0.,  1.,  1., -1.]],
              [[1.,  1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
               [0.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,  0.],
               [1.,  1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
               [0.,  0.,  0.,  1., -1.,  0.,  0.,  0.,  0.,  0.],
               [0.,  0.,  0.,  0.,  0.,  1., -1.,  0.,  0.,  0.],
               [0.,  0.,  0.,  0.,  0.,  0.,  1.,  1., -1., -1.]]]
        sol_c = [np.array([[106.66666667, 106.66666667, 106.66666667],
                           [180.,         180.,         180.]]),
                 np.array([[320., 320., 320.],
                           [60.,  60.,  60.]]),
                 np.array([[203.63636358, 203.63636358, 116.36363642],
                           [90.00000004,  89.99999996, 180.]]),
                 np.array([[94.76075857,  94.76075848, 189.52151705,
                            130.47848295, 130.47848295],
                           [109.87696965, 109.87696965,  70.12303035,
                            89.99999997,  90.00000003]])]
        diagonals = [[0, 1, 2], [2, 1, 0], [1, 0, 2], [2, 0, 4, 1, 3]]
        for i, diag in enumerate(diagonals):
            B, dim = r.do_diagonal_rectangulation(diag)
            self.assertTrue((B == Bc[i]).all())
            dim[0, :] *= w
            dim[1, :] *= h
            E = r.build_rectangulation_equations(B)
            self.assertTrue((E == Ec[i]).all())
            sol = r.minimize_rectangulation(E, dim, w, h, k, c)
            self.assertLess(np.sum(np.abs(sol_c[i] - sol)), 0.01)
            r.draw_resized_rectangles(B, dim, w, h)

    def test_diagonal_rectangulation_15rect(self):
        w = 320
        h = 180
        k = 1.5
        # 0.1: proportion is predominant
        # 0.05: seems good ballanced
        c = 0.05

        B, dim = r.do_diagonal_rectangulation(
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
        r.draw_resized_rectangles(B, dim, 1, 1)
        dim[0, :] *= w
        dim[1, :] *= h
        E = r.build_rectangulation_equations(B)
        mat_sol = r.minimize_rectangulation(E, dim, w, h, k, c)
        r.draw_resized_rectangles(B, mat_sol, w, h)

    def test_diagonal_rectangulation_random(self):
        w = 320
        h = 180
        k = 1.5
        c = 0.05

        rect = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        shuffle(rect)
        print(rect)
        B, dim = r.do_diagonal_rectangulation(rect)
        r.draw_resized_rectangles(B, dim, 1, 1)
        dim[0, :] *= w
        dim[1, :] *= h
        E = r.build_rectangulation_equations(B)
        mat_sol = r.minimize_rectangulation(E, dim, w, h, k, c)
        r.draw_resized_rectangles(B, mat_sol, w, h)

    def test_best_3rect(self):
        N = 3
        c = 0.05
        k = 1.33
        w = 320
        h = 180
        B, sol, seq = r.get_best_rect_for_window(N, c, k, w, h)
        print(sol)
        sol_c = np.array([[116.36363645, 203.63636355, 203.63636355],
                          [180.,          89.99999993,  90.00000007]])
        self.assertLess(np.sum(np.abs(sol_c - sol)), 0.01)
        r.draw_resized_rectangles(B, sol, w, h)

    def test_best_5rect(self):
        N = 5
        c = 0.05
        k = 1.33
        w = 320
        h = 180
        B, sol, seq = r.get_best_rect_for_window(N, c, k, w, h)
        print(sol)
        sol_c = np.array([[159.99999938, 160.00000062, 106.66666643,
                           106.6666668, 106.66666677],
                          [77.17563584, 77.17563584, 102.82436416,
                           102.82436416, 102.82436416]])
        self.assertLess(np.sum(np.abs(sol_c - sol)), 0.01)
        r.draw_resized_rectangles(B, sol, w, h)

    def test_subseq_generation(self):
        subseqs = []
        for ss in r.get_subsequence(3, [1, 2, 3, 4]):
            subseqs.append(ss)
        self.assertEqual([[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]], subseqs)

    def test_baxter_permutation(self):
        self.assertEqual(r.is_baxter_permutation([1, 2, 3, 4]), True)
        self.assertEqual(r.is_baxter_permutation([3, 1, 4, 2]), False)
        self.assertEqual(r.is_baxter_permutation([2, 4, 1, 3]), False)

    def test_count_number_diagonal_rects(self):
        num_rects_for_N = [1, 2, 6, 22, 92, 422, 2074]
        for N in range(1, 8):
            self.assertEqual(num_rects_for_N[N - 1],
                             r.count_number_diagonal_rects(N))

    def test_subseq_match_pattern(self):
        seq = [3, 7, 5, 4, 2, 1, 6]
        pattern = [3, 1, 4, 2]
        gaps = [False, True, False, True]
        self.assertEqual(r.subseq_matches_pattern(seq, [3, 5, 4, 1],
                                                  pattern, gaps), False)
        gaps = [True, True, True, True]
        self.assertEqual(r.subseq_matches_pattern([4, 5, 3, 1, 2],
                                                  [4, 5, 1, 2],
                                                  [3, 4, 1, 2], gaps), True)
        gaps = [True, False, True, False]
        self.assertEqual(r.subseq_matches_pattern([4, 5, 3, 1, 2],
                                                  [4, 5, 1, 2],
                                                  [3, 4, 1, 2], gaps), True)
        gaps = [True, True, False, True]
        self.assertEqual(r.subseq_matches_pattern([4, 5, 3, 1, 2],
                                                  [4, 5, 1, 2],
                                                  [3, 4, 1, 2], gaps), False)

    # Draws best rectangulations while width/height ratio increases.
    def test_best_rect_for_w_h_ratio(self):
        N = 5
        c = 0.05
        num_pt = 15
        # For instance 320x180
        num_pix = 57600
        aspect_lb = 0.3
        aspect_ub = 4.
        # Usual camera x/y ratio
        k = 1.33
        for aspect in np.linspace(aspect_lb, aspect_ub, num_pt):
            w = math.sqrt(aspect*num_pix)
            h = w/aspect
            B, sol, seq = r.get_best_rect_for_window(N, c, k, w, h)
            print(seq)
            title = 'w/h ratio = {:.2f}'.format(aspect)
            r.draw_resized_rectangles(B, sol, w, h, title=title)


if __name__ == '__main__':
    unittest.main()

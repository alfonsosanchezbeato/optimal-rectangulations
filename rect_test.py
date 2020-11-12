import numpy as np
import unittest
import rectangles as r
from scipy.optimize import fsolve


class TestStringMethods(unittest.TestCase):

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

    def _test_diagonal_rectangulation_3rect(self):
        B = r.do_diagonal_rectangulation([0, 1, 2])
        print(B)
        r.draw_rectangles(B, 400, 200)
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

        def dfunc(X): return r.get_derivative_from_eqs(X, E, w, h, k)

        X = fsolve(dfunc, initial_est)
        print(X, r.get_optimization_f_val(X, E, w, h, k))
        print("Using scipy minimize:")
        c = 0.05
        T = c*w*h
        print(r.minimize_rectangulation(E, w, h, k, T))
        print("Using sympy:")
        sol = r.solve_rectangle_eqs(E, w, h, k)
        print(sol)
        mat_sol = self.get_matrix_from_finiteset(3, sol)
        r.draw_resized_rectangles(B, mat_sol, w, h)
        print("Fitting rectangles:")
        print(r.solve_fit_rectangles(E, B, w, h, k))

        B = r.do_diagonal_rectangulation([2, 1, 0])
        print(B)
        r.draw_rectangles(B, 400, 200)
        Bc = np.array([[0, 0, 0],
                       [1, 1, 1],
                       [2, 2, 2]],
                      dtype=int)
        self.assertTrue((B == Bc).all())
        E = r.build_rectangulation_equations(B)
        print(E)
        X = fsolve(dfunc, initial_est)
        print(X, r.get_optimization_f_val(X, E, w, h, k))
        print("Using scipy minimize:")
        print(r.minimize_rectangulation(E, w, h, k, T))
        print("Using sympy:")
        sol = r.solve_rectangle_eqs(E, w, h, k)
        print(sol)
        mat_sol = self.get_matrix_from_finiteset(3, sol)
        r.draw_resized_rectangles(B, mat_sol, w, h)
        print("Fitting rectangles:")
        print(r.solve_fit_rectangles(E, B, w, h, k))
        B = r.do_diagonal_rectangulation([1, 0, 2])
        print(B)
        r.draw_rectangles(B, 400, 200)
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
        print(X, r.get_optimization_f_val(X, E, w, h, k))
        print("Using scipy minimize:")
        print(r.minimize_rectangulation(E, w, h, k, T))
        print("Using sympy:")
        sol = r.solve_rectangle_eqs(E, w, h, k)
        print(sol)
        mat_sol = self.get_matrix_from_finiteset(3, sol)
        r.draw_resized_rectangles(B, mat_sol, w, h)
        print("Fitting rectangles:")
        print(r.solve_fit_rectangles(E, B, w, h, k))

    def _test_diagonal_rectangulation_5rect(self):
        B = r.do_diagonal_rectangulation([2, 0, 4, 1, 3])
        print(B)
        r.draw_rectangles(B, 400, 200)
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
        c = 0.05
        T = c*w*h
        print("Using scipy minimize:")
        print(r.minimize_rectangulation(E, w, h, k, T))
        print("Using sympy:")
        sol = r.solve_rectangle_eqs(E, w, h, k)
        print(sol)
        mat_sol = self.get_matrix_from_finiteset(5, sol)
        r.draw_resized_rectangles(B, mat_sol, w, h)
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
        w = 400
        h = 200
        k = 1.5
        r.draw_rectangles(B, w, h)
        E = r.build_rectangulation_equations(B)

        print("Using scipy minimize:")
        # 0.1: proportion is predominant
        # 0.05: ?
        T = 0.05*w*h
        mat_sol = r.minimize_rectangulation(E, w, h, k, T)
        print(mat_sol)
        vals = mat_sol[0, :].tolist()
        vals.extend(mat_sol[1, :].tolist())
        print(r.opt_f_val(vals, w, h, k, T))

        # print("Using sympy:")
        # sol = r.solve_rectangle_eqs(E, w, h, k)
        # print(sol)
        # mat_sol = self.get_matrix_from_finiteset(15, sol)
        r.draw_resized_rectangles(B, mat_sol, w, h)

        print(E)

    def _test_best_3rect(self):
        N = 3
        k = 1.5
        w = 400
        h = 200
        B, sol = r.get_best_rect_for_window(N, k, w, h)
        if sol.size:
            print("Best solution is", sol)
            r.draw_resized_rectangles(B, sol, w, h)
        else:
            print("No solution found")

    def _test_best_5rect(self):
        N = 5
        k = 1.5
        w = 400
        h = 200
        B, sol = r.get_best_rect_for_window(N, k, w, h)
        if sol.size:
            print("Best solution is", sol)
            r.draw_resized_rectangles(B, sol, w, h)
        else:
            print("No solution found")

    def test_subseq_match_pattern(self):
        seq = [3, 7, 5, 4, 2, 1, 6]
        pattern = [3, 1, 4, 2]
        gaps = [False, True, False, True]
        print(r.does_subseq_match_pattern(seq, [3, 5, 4, 1], pattern, gaps))
        gaps = [True, True, True, True]
        print(r.does_subseq_match_pattern([4,5,3,1,2], [4,5,1,2], [3,4,1,2], gaps))
        gaps = [True, False, True, False]
        print(r.does_subseq_match_pattern([4,5,3,1,2], [4,5,1,2], [3,4,1,2], gaps))
        gaps = [True, True, False, True]
        print(r.does_subseq_match_pattern([4,5,3,1,2], [4,5,1,2], [3,4,1,2], gaps))


if __name__ == '__main__':
    unittest.main()

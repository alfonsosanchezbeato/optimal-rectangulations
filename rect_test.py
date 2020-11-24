import numpy as np
import unittest
import rectangles as r
import rect_lagrange as rl
import rect_fit_sides as rfs
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

    def test_diagonal_rectangulation_3rect(self):
        w = 320
        h = 180
        k = 1.5
        c = 0.05
        B, dim = r.do_diagonal_rectangulation([0, 1, 2])
        print(B)
        dim[0, :] *= w
        dim[1, :] *= h
        r.draw_resized_rectangles(B, dim, w, h)
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
        print("Using scipy minimize:")
        print(r.minimize_rectangulation(E, dim, w, h, k, c))

        B, dim = r.do_diagonal_rectangulation([2, 1, 0])
        print(B)
        dim[0, :] *= w
        dim[1, :] *= h
        r.draw_resized_rectangles(B, dim, w, h)
        Bc = np.array([[0, 0, 0],
                       [1, 1, 1],
                       [2, 2, 2]],
                      dtype=int)
        self.assertTrue((B == Bc).all())
        E = r.build_rectangulation_equations(B)
        print(E)
        print("Using scipy minimize:")
        print(r.minimize_rectangulation(E, dim, w, h, k, c))
        B, dim = r.do_diagonal_rectangulation([1, 0, 2])
        print(B)
        dim[0, :] *= w
        dim[1, :] *= h
        Bc = np.array([[0, 0, 2],
                       [1, 1, 2],
                       [1, 1, 2]],
                      dtype=int)
        self.assertTrue((B == Bc).all())
        E = r.build_rectangulation_equations(B)
        print(E)
        print("Using scipy minimize:")
        print(r.minimize_rectangulation(E, dim, w, h, k, c))

    def test_diagonal_rectangulation_5rect(self):
        w = 320
        h = 180
        k = 1.5
        c = 0.05

        B, dim = r.do_diagonal_rectangulation([2, 0, 4, 1, 3])
        print(B)
        dim[0, :] *= w
        dim[1, :] *= h
        print(dim)
        r.draw_resized_rectangles(B, dim, w, h)
        Bc = np.array([[0, 1, 1, 3, 3],
                       [0, 1, 1, 3, 3],
                       [2, 2, 2, 3, 3],
                       [2, 2, 2, 3, 3],
                       [2, 2, 2, 4, 4]],
                      dtype=int)
        self.assertTrue((B == Bc).all())
        E = r.build_rectangulation_equations(B)
        print(E)
        print("Using scipy minimize:")
        print(r.minimize_rectangulation(E, dim, w, h, k, c))

    def test_diagonal_rectangulation_15rect(self):
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
        w = 320
        h = 180
        k = 1.5
        dim[0, :] *= w
        dim[1, :] *= h
        r.draw_resized_rectangles(B, dim, w, h)

        E = r.build_rectangulation_equations(B)

        print("Using scipy minimize:")
        # 0.1: proportion is predominant
        # 0.05: ?
        c = 0.05
        mat_sol = r.minimize_rectangulation(E, dim, w, h, k, c)
        print(mat_sol)
        vals = mat_sol[0, :].tolist()
        vals.extend(mat_sol[1, :].tolist())
        print(r.opt_f_val(vals, w, h, k, c))

        r.draw_resized_rectangles(B, mat_sol, w, h)

        print(E)

    def test_best_3rect(self):
        N = 3
        k = 1.5
        w = 320
        h = 180
        B, sol, seq = r.get_best_rect_for_window(N, k, w, h)
        if sol.size:
            print("Best solution is", sol)
            r.draw_resized_rectangles(B, sol, w, h)
        else:
            print("No solution found")

    def test_best_5rect(self):
        N = 5
        k = 1.5
        w = 320
        h = 180
        B, sol, seq = r.get_best_rect_for_window(N, k, w, h)
        if sol.size:
            print("Best solution is", sol)
            r.draw_resized_rectangles(B, sol, w, h)
        else:
            print("No solution found")

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
        print(r.subseq_matches_pattern(seq, [3, 5, 4, 1], pattern, gaps))
        gaps = [True, True, True, True]
        print(r.subseq_matches_pattern([4, 5, 3, 1, 2], [4, 5, 1, 2],
                                       [3, 4, 1, 2], gaps))
        gaps = [True, False, True, False]
        print(r.subseq_matches_pattern([4, 5, 3, 1, 2], [4, 5, 1, 2],
                                       [3, 4, 1, 2], gaps))
        gaps = [True, True, False, True]
        print(r.subseq_matches_pattern([4, 5, 3, 1, 2], [4, 5, 1, 2],
                                       [3, 4, 1, 2], gaps))


if __name__ == '__main__':
    unittest.main()

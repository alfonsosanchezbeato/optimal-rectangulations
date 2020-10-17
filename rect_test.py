import numpy as np
import unittest
import rectangles as r


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
        B = r.do_diagonal_rectangulation([2, 1, 0])
        print(B)
        Bc = np.array([[0, 0, 0],
                       [1, 1, 1],
                       [2, 2, 2]],
                      dtype=int)
        self.assertTrue((B == Bc).all())
        B = r.do_diagonal_rectangulation([1, 0, 2])
        print(B)
        Bc = np.array([[0, 0, 2],
                       [1, 1, 2],
                       [1, 1, 2]],
                      dtype=int)
        self.assertTrue((B == Bc).all())

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

    def test_diagonal_rectangulation_15rect(self):
        B = r.do_diagonal_rectangulation(
            [7, 12, 6, 4, 10, 1, 13, 5, 14, 8, 9, 2, 0, 3, 11])
        print(B)
        Bc = np.array([[0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 11, 11, 11, 11],
                       [1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 11, 11, 11, 11],
                       [1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 11, 11, 11, 11],
                       [1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 11, 11, 11, 11],
                       [4, 4, 4, 4, 4, 5, 5, 5, 8, 9, 9, 11, 11, 11, 11],
                       [4, 4, 4, 4, 4, 5, 5, 5, 8, 9, 9, 11, 11, 11, 11],
                       [6, 6, 6, 6, 6, 6, 6, 6, 8, 9, 9, 11, 11, 11, 11],
                       [7, 7, 7, 7, 7, 7, 7, 7, 8, 9, 9, 11, 11, 11, 11],
                       [7, 7, 7, 7, 7, 7, 7, 7, 8, 9, 9, 11, 11, 11, 11],
                       [7, 7, 7, 7, 7, 7, 7, 7, 8, 9, 9, 11, 11, 11, 11],
                       [7, 7, 7, 7, 7, 7, 7, 7, 10, 10, 10, 11, 11, 11, 11],
                       [7, 7, 7, 7, 7, 7, 7, 7, 10, 10, 10, 11, 11, 11, 11],
                       [7, 7, 7, 7, 7, 7, 7, 7, 12, 12, 12, 12, 12, 13, 14],
                       [7, 7, 7, 7, 7, 7, 7, 7, 12, 12, 12, 12, 12, 13, 14],
                       [7, 7, 7, 7, 7, 7, 7, 7, 12, 12, 12, 12, 12, 13, 14]],
                      dtype=int)
        self.assertTrue((B == Bc).all())


if __name__ == '__main__':
    unittest.main()

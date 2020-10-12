import numpy as np
import unittest
from copy import deepcopy
from fractions import Fraction
import rectangles as r


class TestStringMethods(unittest.TestCase):

    def print_rects(self, R):
        print('\nRectangles:')
        for ri in R:
            print(ri)

    def test_split_rectangle_vertically(self):
        N = 3
        R = [r.RectangleLayout(Fraction(0, 1), Fraction(0, 1),
                               Fraction(1, 1), Fraction(1, 1))]
        E = np.zeros([2, 2*N])
        # Initial equations w1 = 1, w2 = 1
        E[0, 0] = 1
        E[1, N] = 1

        R, E = r.split_rectangle_vertically(0, N, R, E)
        self.print_rects(R)
        Rc = [r.RectangleLayout(Fraction(0, 1), Fraction(0, 1),
                                Fraction(1, 2), Fraction(1, 1)),
              r.RectangleLayout(Fraction(1, 2), Fraction(0, 1),
                                Fraction(1, 2), Fraction(1, 1))]
        Ec = np.array([[1, 1, 0, 0,  0, 0],
                       [0, 0, 0, 1,  0, 0],
                       [0, 0, 0, 1, -1, 0]])
        self.assertTrue(R == Rc)
        self.assertTrue((E == Ec).all())

        R1 = deepcopy(R)
        E1 = deepcopy(E)
        R, E = r.split_rectangle_vertically(0, N, R, E)
        self.print_rects(R)
        Rc = [r.RectangleLayout(Fraction(0, 1), Fraction(0, 1),
                                Fraction(1, 4), Fraction(1, 1)),
              r.RectangleLayout(Fraction(1, 2), Fraction(0, 1),
                                Fraction(1, 2), Fraction(1, 1)),
              r.RectangleLayout(Fraction(1, 4), Fraction(0, 1),
                                Fraction(1, 4), Fraction(1, 1))]
        Ec = np.array([[1, 1, 1, 0,  0,  0],
                       [0, 0, 0, 1,  0,  0],
                       [0, 0, 0, 0, -1,  1],
                       [0, 0, 0, 1,  0, -1]])
        self.assertTrue(R == Rc)
        self.assertTrue((E == Ec).all())

        R1, E1 = r.split_rectangle_vertically(1, N, R1, E1)
        self.print_rects(R)
        Rc = [r.RectangleLayout(Fraction(0, 1), Fraction(0, 1),
                                Fraction(1, 2), Fraction(1, 1)),
              r.RectangleLayout(Fraction(1, 2), Fraction(0, 1),
                                Fraction(1, 4), Fraction(1, 1)),
              r.RectangleLayout(Fraction(3, 4), Fraction(0, 1),
                                Fraction(1, 4), Fraction(1, 1))]
        Ec = np.array([[1, 1, 1, 0,  0,  0],
                       [0, 0, 0, 1,  0,  0],
                       [0, 0, 0, 1, -1,  0],
                       [0, 0, 0, 0,  1, -1]])
        self.assertTrue(R1 == Rc)
        self.assertTrue((E1 == Ec).all())


if __name__ == '__main__':
    unittest.main()

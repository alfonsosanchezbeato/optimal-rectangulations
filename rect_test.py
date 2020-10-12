import unittest
from fractions import Fraction
import rectangles as r


class TestStringMethods(unittest.TestCase):

    def test_background_corners(self):
        B = r.RectangleLayout(Fraction(0, 1), Fraction(0, 1),
                              Fraction(1, 1), Fraction(1, 1))
        R = []
        x, y = r.get_topleft_corner(B, R)
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)
        B.top = [r.RectangleLayout(Fraction(0, 1), Fraction(0, 1),
                                   Fraction(1, 2), Fraction(1, 2))]
        x, y = r.get_topleft_corner(B, R)
        self.assertEqual(x, Fraction(1, 2))
        self.assertEqual(y, 0)
        B.top = [r.RectangleLayout(Fraction(0, 1), Fraction(0, 1),
                                   Fraction(1, 1), Fraction(1, 2))]
        B.left = B.top
        B.rigth = B.top
        x, y = r.get_topleft_corner(B, R)
        self.assertEqual(x, 0)
        self.assertEqual(y, Fraction(1, 2))
        B.top = [r.RectangleLayout(Fraction(0, 1), Fraction(0, 1),
                                   Fraction(1, 1), Fraction(1, 1))]
        B.left = B.top
        B.rigth = B.top
        B.bottom = B.top
        x, y = r.get_topleft_corner(B, R)
        self.assertEqual(x, -1)
        self.assertEqual(y, -1)


if __name__ == '__main__':
    unittest.main()

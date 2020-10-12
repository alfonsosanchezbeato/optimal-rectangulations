#!/usr/bin/env python3
import numpy as np
from fractions import Fraction


# class RectLimitSegment:
#     def __init__(self, rect, ):


class RectangleLayout:
    def __init__(self, x, y, width, height):
        # Fractions that define relative positions in the background
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        # Lists with nearby rectangles
        self.top = []
        self.right = []
        self.bottom = []
        self.left = []

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

    # Is the side "closed"? (no space for more rectangles)

    def top_closed(self):
        if self.top and self.top[-1].max_x() >= self.max_x():
            return True
        return False

    def right_closed(self):
        if self.right and self.top[-1].max_y() >= self.max_y():
            return True
        return False

    def bottom_closed(self):
        if self.bottom and self.bottom[-1].max_x() >= self.max_x():
            return True
        return False

    def left_closed(self):
        if self.left and self.left[-1].max_y() >= self.max_y():
            return True
        return False


def get_rects_corner(R):
    x = Fraction(-1, 1)
    y = Fraction(-1, 1)
    for r in R:
        # Check first if space left to the right, then if
        # 1. No space, go to next (if corner we will find out later
        #    when looking at rectangles at bottom)
        # 2. Already a right rectangle, but some space: we
        #    know then the two rectangles that form the corner
        # 3. Empty: go to top, then if
        #    1. Top not closed, no corner, go to next
        #    2. Top closed, look at last rectangle there
        #       1. Last rectangle expands further to the right: corner
        #       2. Otherwise, look at its right. If closed, corner.
        if r.right_closed():
            continue
        if r.right:
            x = r.max_x()
            y = r.right[-1].max_y()
            break
        # Empty right, check top
        if not r.top_closed():
            continue
        top_r = r.top[-1]
        if top_r.max_x() > r.max_x():
            x = r.max_x()
            y = r.y
            break
        if top_r.right_closed():
            x = r.max_x()
            y = r.y
            break

    return x, y


def get_topleft_corner(B, R):
    x = Fraction(-1, 1)
    y = Fraction(-1, 1)
    # Top of B
    if not B.top_closed():
        if not B.top:
            x = Fraction(0, 1)
        else:
            x = B.top[-1].max_x()
        y = Fraction(0, 1)
    elif not B.left_closed():
        # Left of B
        x = Fraction(0, 1)
        if not B.left:
            y = Fraction(0, 1)
        else:
            y = B.left[-1].max_y()
    else:
        # Corners limited by regular rectangles
        x, y = get_rects_corner(R)

    return x, y


def _split_rectangle():
    # Background rectangle
    B = RectangleLayout(Fraction(0, 1), Fraction(0, 1),
                        Fraction(1, 1), Fraction(1, 1))
    # Rectangles list
    R = []
    x, y = get_topleft_corner(B, R)
    print("Next corner:", x, ",", y)


# r_idx: index to rectangle to be split
# N: final number of rectangles
# R: current list of rectangles, a new one will be appended
# E: list of equations, it will be modified
def split_rectangle_vertically(r_idx, N, R, E):
    # Modify r and create a new rectangle
    r = R[r_idx]
    r.width /= 2
    r2 = RectangleLayout(r.x + r.width, r.y, r.width, r.height)
    index = len(R)
    R.append(r2)
    # Modify equations appropriately
    # w_old -> w_1 + w_2
    for eq in E:
        if eq[r_idx] != 0:
            eq[index] = eq[r_idx]
    # h_old -> h_2 iff line to the right of r_old
    for eq in E:
        v_1 = eq[N + r_idx]
        if v_1 != 0:
            # Different sign implies rectange on different side to r
            for i, v in enumerate(eq[N:]):
                # We only have 0, 1 or -1 as possible values
                if v != 0 and v != v_1 and R[r_idx].x < R[i].x:
                    eq[N + index] = v_1
                    eq[N + r_idx] = 0
    # h_1 - h_2 = 0
    new_eq = np.zeros(2*N)
    new_eq[N + r_idx] = 1
    new_eq[N + index] = -1
    E = np.vstack([E, new_eq])
    return R, E


# Data modelling.
# N: final number of squares
# Equations contain 2*N variables: w1,..,wN,h1,..,hN
# Final number of equations will be N+1

if __name__ == '__main__':
    R = [RectangleLayout(Fraction(0, 1), Fraction(0, 1),
                         Fraction(1, 1), Fraction(1, 1))]
    _split_rectangle()

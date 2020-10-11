#!/usr/bin/env python3
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


def split_rectangle():
    # Background rectangle
    B = RectangleLayout(Fraction(0, 1), Fraction(0, 1),
                        Fraction(1, 1), Fraction(1, 1))
    # Rectangles list
    R = []
    x, y = get_topleft_corner(B, R)
    print("Next corner:", x, ",", y)


if __name__ == '__main__':
    split_rectangle()

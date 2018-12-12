from pylab import *
import scipy.stats as stats
import numpy as np

def intersect(a1, a2, b1, b2):
    """
    checks if the line from a1 to a2 and the line from b1 to b2 cross in 2D
    :param a1: start point first line [a1_x,a1_y]
    :param a2: end point fist line [a2_x,a2_y]
    :param b1: start point second line [b1_x,b1_y]
    :param b2: end point second line [b2_x,b2_y]
    """

    min_ax, max_ax = sorted([a1[0], a2[0]])
    min_ay ,max_ay = sorted([a1[1], a2[1]])
    min_bx, max_bx = sorted([b1[0], b2[0]])
    min_by, max_by = sorted([b1[1], b2[1]])


    min_ax = min([a1[0], a2[0]])
    max_ax = max([a1[0], a2[0]])
    min_ay = min([a1[1], a2[1]])
    max_ay = max([a1[1], a2[1]])
    min_bx = min([b1[0], b2[0]])
    max_bx = max([b1[0], b2[0]])
    min_by = min([b1[1], b2[1]])
    max_by = max([b1[1], b2[1]])

    try:
        inter_x = ((a1[0] * a2[1] - a1[1] * a2[0]) * (b1[0] - b2[0]) - (a1[0] - a2[0]) * (b1[0] * b2[1] - b1[1] * b2[0])) / ((a1[0] - a2[0]) * (b1[1] - b2[1]) - (a1[1] - a2[1]) * (b1[0] - b2[0]))
        inter_y = ((a1[0] * a2[1] - a1[1] * a2[0]) * (b1[1] - b2[1]) - (a1[1] - a2[1]) * (b1[0] * b2[1] - b1[1] * b2[0])) / ((a1[0] - a2[0]) * (b1[1] - b2[1]) - (a1[1] - a2[1]) * (b1[0] - b2[0]))
        inter_x = round(inter_x, 5)
        inter_y = round(inter_y, 5)
    except:
        return False
    if min_ax <= inter_x <= max_ax and min_ay <= inter_y <= max_ay and min_bx <= inter_x <= max_bx and min_by <= inter_y <= max_by:
        return[inter_x,inter_y]

#!/usr/bin/env python
from pylab import *
import scipy.stats as stats
import numpy as np
#

p1 = [0,0]
p2 = [2,2]
p3 = [2,0]
p4 = [0,2]
def intersect(p1,p2,p3,p4):
    listx1 = [p1[0],p2[0]]
    listy1 = [p1[1],p2[1]]
    minx1 = min(listx1)
    maxx1 = max(listx1)
    miny1 = min(listy1)
    maxy1 = max(listy1)
    listx2 = [p3[0], p4[0]]
    listy2 = [p3[1], p4[1]]
    minx2 = min(listx2)
    maxx2 = max(listx2)
    miny2 = min(listy2)
    maxy2 = max(listy2)
    px =((p1[0] * p2[1] - p1[1] * p2[0]) * (p3[0] - p4[0]) - (p1[0] - p2[0]) * (p3[0] * p4[1] - p3[1] * p4[0])) / ((p1[0] - p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] - p4[0]))
    py =((p1[0] * p2[1] - p1[0] * p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] * p4[1] - p3[1] * p4[0])) / ((p1[0] - p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] - p4[0]))
    if minx1 <= px <= maxx1 and miny1 <= py <= maxy1 and minx2 <= px <= maxx2 and miny2 <= py <= maxy2:
        return[px,py]


print(intersect(p1,p2,p3,p4))
x = intersect(p1,p2,p3,p4)


scatter(p1[0], p1[1])
scatter(p2[0],p2[1])
scatter(p3[0],p3[1])
scatter(p4[0],p4[1])
scatter(x[0],x[1])

show()
import numpy as np
import matplotlib.pyplot as plt


class waypoint:
    num = 0
    ''' class object of a waypoint, has attribute of position '''

    def __init__(self, xy):
        self.position = xy
        self.id = self.num + 1
        waypoint.num = self.id


def decision(probabilities):
    # quinten
    choice = 0
    return choice


def main():
    t = 0
    while t < 100:
        break


######## initial
# for nora (travelling salesman)
def random_waypoints(n=10, set_seed=True):
    if set_seed:
        np.random.seed(0)  # to get each time the same random numbers
    points = []
    for i in range(n):
        pos = np.random.rand(2)
        points.append(waypoint(pos))
    return points

def initial_condition(waypoints):
    return waypoints, np.random.shuffle(waypoints)


########


if __name__ == '__main__':
    points, config = initial_condition()
    main()

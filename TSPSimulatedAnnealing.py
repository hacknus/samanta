import numpy as np
import matplotlib.pyplot as plt


class Waypoint:
    num = 0
    ''' class object of a waypoint, has attribute of position '''

    def __init__(self, xy):
        self.position = xy
        self.id = self.num + 1
        Waypoint.num = self.id

def temp_0(waypoints):
    """
estimates a good starting temperature
    :param waypoints: used waypoints
    :return start temperatur
    """
    pass


def dist_array(waypoints):
    """
calculates distances between waypoints
    :param waypoints: list of waypoints
    :return dist_array: n x n array, with dist_array[i][j] = distance between ith and jth waypoint
    """
    n = len(waypoints)
    dist = [
        [np.sqrt(
            (waypoints[i].position[0] - waypoints[j].position[0]) ** 2 +
            (waypoints[i].position[1] - waypoints[j].position[1]) ** 2
        ) for j in range(n)]
        for i in range(n)]
    return dist


def cost(config, dist):
    """
calculates cost resp. distance for the configuration
    """
    cost = sum([dist[i][(i + 1) % len(config)] for i in config])
    return cost


def acceptance_probability(config_k, config_k1, tempk, dist):
    """
Determines probability of accepting the new config
    :param config_k: config at time k
    :param config_k1: config at time k+1
    :param tempk: temp at time k
    :param dist: distance array for used waypoints
    :return p: probability of acceptance
    """
    delta = cost(config_k1, dist) - cost(config_k, dist)
    if delta <= 0:
        p = 1
    else:
        p = np.exp(-delta / tempk)
    return p


def change_config1(config):
    """
randomly changes configuration: e.g.: a--b--c--d--e--f ==> a--e--d--c--b--f
    arg: old config
    return: new config
    """

    return config


def change_config2(config):
    """
randomly inserts a part of the config somewhere else, e.g. a--b--c--d--e--f ==> a--d--e--b--c--f
    :param config: old config
    :return: new config
    """
    return config

def move(p):
    """
Decides if to accept the new config
    :param p: probability
    :return: boolean, if accepted ==> True, else ==> False
    """
    if np.random.rand() < p:
        return True
    else:
        return False


def main():
    t = 0
    while t < 100:
        break


# ----------------- initial--------------
# for nora (travelling salesman)
def random_waypoints(n=10, set_seed=True):
    if set_seed:
        np.random.seed(0)  # to get each time the same random numbers
    points = [Waypoint(np.random.rand(2)) for i in range(n)]
    points_dict = {}
    for p in points:
        points_dict[p.id] = p
    return points_dict


def initial_condition(waypoints):
    return waypoints, [i for i in range(len(waypoints))]


# --------------------


if __name__ == '__main__':
    points, config = initial_condition(random_waypoints())

    main()

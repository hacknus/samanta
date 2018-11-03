import numpy as np
import matplotlib.pyplot as plt


class Waypoint:
    num = 0
    ''' class object of a waypoint, has attribute of position '''

    def __init__(self, xy):
        self.position = xy
        self.id = self.num + 1
        Waypoint.num = self.id


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


def cost(config):
    """
calculates cost resp. distance for the configuration
    """
    pass


def acceptance_probability(config_k, config_k1, tempk):
    """
Determines probability of accepting the new config
    :param config_k: config at time k
    :param config_k1: config at time k+1
    :param temp: temp at time k
    :return p: probability of acceptance
    """
    delta = cost(config_k1) - cost(config_k)
    if delta <= 0:
        p = 1
    else:
        p = np.exp(-delta / tempk)
    return p


def decision(p):
    pass


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
    return points


def initial_condition(waypoints):
    return waypoints, np.random.shuffle(waypoints)


# --------------------


if __name__ == '__main__':
    points, config = initial_condition(random_waypoints())
    main()

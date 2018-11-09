import numpy as np
import matplotlib.pyplot as plt


class Cities:
    num = 0
    ''' class object of a waypoint, has attribute of position '''

    def __init__(self, xy):
        self.position = xy
        self.id = Cities.num
        Cities.num += 1

    def __repr__(self):
        return str(self.id)


def dist_array(cities):
    """
calculates distances between cities
maybe np.linalg.norm
    :param cities: list of cities objects
    :return dist_array: n x n array, with dist_array[i][j] = distance between ith and jth city
    """
    n = len(cities)
    dist = [
        [np.sqrt(
            (cities[i].position[0] - cities[j].position[0]) ** 2 +
            (cities[i].position[1] - cities[j].position[1]) ** 2
        ) for j in range(n)]
        for i in range(n)]
    return dist


def swap2cities(config):
    """
swaps two cities in configuration
    :param config: old configuration
    :return: new configuration
    """
    i, j = np.random.randint(len(config), size=2)
    new_config = np.copy(config)
    temp = new_config[i]
    new_config[i] = new_config[j]
    new_config[j] = temp
    return new_config


# def change_config1(config):
#     """
# randomly changes configuration: e.g.: a--b--c--d--e--f ==> a--e--d--c--b--f
#     arg: old config
#     return: new config
#     """
#
#     return config
#
#
# def change_config2(config):
#     """
# randomly inserts a part of the config somewhere else, e.g. a--b--c--d--e--f ==> a--d--e--b--c--f
#     :param config: old config
#     :return: new config
#     """
#     return config


def cost(config, dist):
    """
calculates cost resp. distance for the configuration
    """
    cost = np.sum([dist[config[i]][config[(i + 1) % len(config)]] for i in range(len(config))])
    return cost


def acceptance_probability(config_k, config_k1, tempk, dist):
    """
Determines probability of accepting the new config
    :param config_k: config at time k
    :param config_k1: config at time k+1
    :param tempk: temp at time k
    :param dist: distance array for used cities
    :return p: probability of acceptance
    """
    delta = cost(config_k1, dist) - cost(config_k, dist)
    if delta <= 0:
        p = 1
    else:
        p = np.exp(-delta / tempk)
    return p


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


def temp_0(dist):
    """
estimates a good starting temperature
    :param dist: distance matrix for used cities
    :return start temperatur
    """
    random_config = [np.random.permutation(np.arange(10)) for i in range(100)]
    cost_random_config = [cost(c, dist) for c in random_config]
    t_0 = (max(cost_random_config) - min(cost_random_config)) * 1.1
    return t_0


# ----------------- initial--------------

def random_cities(n=10, set_seed=True):
    if set_seed:
        np.random.seed(0)  # to get each time the same random numbers
    points = [Cities(np.random.rand(2)) for i in range(n)]
    points_dict = {}
    for p in points:
        points_dict[p.id] = p
    return points


def initial_condition(cities, beta):
    """
choose random state, calculate distance matrix, define t_0, beta
    :param cities:
    :param beta: cooling factor
    :return:
    """
    dist = dist_array(cities)
    t_0 = temp_0(dist)
    list_id = [city.id for city in cities]
    start_config = np.random.permutation(list_id)
    return cities, start_config, dist, beta, t_0


# --------------------

def run(config, t_k, dist, beta):
    """
-create new state s'
-compute delta
    decide if move to new stat
        if yes: compute t_k+1 = beta* t_k
    :param config: current configuration
    :param t_k: current temperatur
    :param dist: distance matrix for used cities
    :param beta: cooling factor
    """
    config_accept = False
    while not config_accept:
        new_config = swap2cities(config)
        p = acceptance_probability(config, new_config, t_k, dist)
        if move(p):
            t_k1 = t_k * beta
            config_accept = True

    return new_config, t_k1


# TestSquare-----------------------------------
# cities1 = [Cities([0,0]), Cities([0,1]), Cities([1,1]), Cities([1,0])]
# dist1 = dist_array(cities1)
# cost1 = cost([0,2,3,1],dist1)
# print('cost1=',cost1)
# print('dist1=',dist1)
# -------------------------------------------------------------

cities = random_cities(20)
cities_dict = {}
for city in cities:
    cities_dict[city.id] = city

x_cities = [c.position[0] for c in cities]
y_cities = [c.position[1] for c in cities]

dist = dist_array(cities)
t_0 = temp_0(dist)
t_min = 1e-6
list_id = [city.id for city in cities]
start_config = np.random.permutation(list_id)
beta = 0.9995

new_config, tk1 = run(start_config, t_0, dist, beta)


def simulated_annealing(start_config, dist, t_0, t_min):
    accepted_configs = [start_config]
    accepted_temps = [t_0]
    best_config = start_config
    best_cost = cost(start_config, dist)

    t = t_0
    config = start_config

    counter = 0
    while t > t_min:
        counter += 1
        config, t = run(config, t, dist, beta)
        accepted_configs.append(config)
        accepted_temps.append(t)

        if cost(config, dist) < best_cost:
            best_config = config
            best_cost = cost(config, dist)

        if counter % 50000 == 0:
            if accepted_configs[-1].all() == best_config.all():
                break
            counter = 0

    return best_config, best_cost, accepted_configs, accepted_temps



best_config, best_cost, accepted_configs, accepted_temps = simulated_annealing(start_config,dist,t_0,t_min)


plt.figure()
fig, ax = plt.subplots(1, 3)
for i in range(3):
    ax[i].scatter(x_cities, y_cities)

x_start_config = [cities_dict[id].position[0] for id in start_config]
x_start_config.append(x_start_config[0])
y_start_config = [cities_dict[id].position[1] for id in start_config]
y_start_config.append(y_start_config[0])

x_new_config = [cities_dict[id].position[0] for id in new_config]
x_new_config.append(x_new_config[0])
y_new_config = [cities_dict[id].position[1] for id in new_config]
y_new_config.append(y_new_config[0])

x_best_config = [cities_dict[id].position[0] for id in best_config]
x_best_config.append(x_best_config[0])
y_best_config = [cities_dict[id].position[1] for id in best_config]
y_best_config.append(y_best_config[0])

ax[0].plot(x_start_config, y_start_config)
ax[1].plot(x_new_config, y_new_config)
ax[2].plot(x_best_config, y_best_config)
print(len(accepted_configs))
plt.show()
#
# #3)repeat step 1, 2, keeping track of best solution until stoping cond
#
# # Animation------------------
# n = 100
# fig = plt.figure()
# ax = plt.axes(xlim=(0, 1), ylim=(0, 1))
# line, = ax.plot([], [], animated=True, lw=1)
# points = np.random.rand(n, 2)
# ax.scatter(points[:, 0], points[:, 1], color='orange')
#
#
# def init():
#     line.set_data([], [])
#     return line,
#
#
# def animate(i):
#     """"""
#     config = np.arange(n)
#     np.random.shuffle(config)
#     """"""
#
#     xdata = points[config, 0]
#     ydata = points[config, 1]
#     line.set_data(xdata, ydata)
#     return line,
#
#
# # ax.scatter(xdata, ydata, s=200, color='black', zorder=1)
# # ax.scatter(xdata, ydata, s=200, color='orange', zorder=1)
# # ax.scatter(xdata, ydata, s=2, color='red', zorder=1)
#
#
# ani = animation.FuncAnimation(fig, animate, np.arange(0, 20), blit=True, interval=20,
#                               repeat=False, init_func=init)
# plt.show()
# # Quinten--------------
#
#
# if __name__ == '__main__':
#     points, config = initial_condition(random_cities())

# command = "{ffmpeg} -r {fps} -i {tmp_dir}/{qty}_image_%05d.png -vcodec mpeg4 -q:v 1 -y {out}".format(ffmpeg='C:\\FFmpeg',fps=10,tmp_dir=tmp_dir,qty=kwargs['qty'],out=outname)
# subprocess.call(command,shell=True)
#
# fps feeds per second
# tmp_dir = ordner mit bilder

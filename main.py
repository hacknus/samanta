import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


rho = 0.99  #evaporation coefficient
Q = 1		#constant quantity


class ant:
	''' ant object, has attributes of position'''
	def __init__(self,xy):
		self.position = xy
		self.tabu = []
		self.last_path = None


class waypoint:
	num = 0
	''' class object of a waypoint, has attribute of position '''
	def __init__(self,xy):
		self.position = xy
		self.id = self.num + 1
		waypoint.num = self.id

def delta(k,p,i,j):
	'''
	takes k-th ant, point list and indices i and j as input
	and returns value of 
	delta tau
	'''
	if k.last_path == [ p[i],p[j] ] or k.last_path == [ p[j],p[i] ]:
		return Q/np.linalg.norm(k.last_path[0]-k.last_path[1])
	else:
		return 0


def update_feromone(f,ant_list,point_list):
	''' takes feromone matrix, ant list and point list as input and'''
	#linus (1. method ant quantity)
	#could be vectorized


	f_new = np.copy(f)
	for i in range(len(f)):
		for j in range(len(f)):
			s = 0
			for k in ant_list:
				s += f[i][j]*rho + delta(k,point_list,i,j)
			f_new[i][j] = s
	return f_new

def decision(probabilities):
	#quinten
	choice = 0
	return choice

def prob(feromone,d):
	''' takes feromone strength and distance to next point as input 
		returns the probability of choosing this path'''
	p = 1
	return p


def init():
	#used to animate later on
    line.set_ydata(np.ma.array(x, mask=True))
    return line,


def animate(i):
	#used to animate later on
    line.set_ydata(np.sin(x + i/10.0))  # update the data
    return line,

######## initial
#for nora (travelling salesman)

def initial_condition(n=20,set_seed=True):
	'''
	takes number of waypoints and a boolean as inputs
	if boolean is true, then the seed will be fixed such that on every run the 'random' values are the same
	this makes it easy to compare in the debug phase
	'''
	Apos = [0,0]	# point A
	Bpos = [1,1]	# point B

	if set_seed:
		np.random.seed(0)			#to get each time the same random numbers
	points = []
	ants = []
	for i in range(n-2):
		pos = np.random.rand(2)
		points.append(waypoint(pos))
		ants.append(ant(pos))
	A = waypoint(Apos)
	B = waypoint(Bpos)
	points.append(A)
	ants.append(ant(Apos))
	points.append(B)
	ants.append(ant(Bpos))
	feromone = np.ones((n,n)) #initialize feromone at t0
	return np.array(points),np.array(ants),feromone

########

def draw_all_paths(p,feromone,paths=[]):
	'''
	takes waypoints and feromone as inputs, optionally previously drawn paths
	this function then draws all paths according to the feromone level
	and returns a list of all matplotlib line-objects (so they can be deleted later on)
	'''
	x = []
	y = []
	if paths:
		for line in paths:
			#clear all previously drawn paths (saves memory)
			line.pop(0).remove()
	for i in range(len(p)):
		for j in range(len(p)):
			if i!=j:
				width = feromone[i][j]/10 #adjusts with according to feromone strength
				paths.append(ax.plot( [p[i][0], p[j][0]] ,[p[i][1], p[j][1]] ,color='blue',linewidth=width,zorder=-2))
	return paths





if __name__ == '__main__':
	fig, ax = plt.subplots()
	#x = np.arange(0, 2*np.pi, 0.01)
	#line, = ax.plot(x, np.sin(x))

	points,ants,feromone = initial_condition()


	#get list of coordinates for plotting purposes
	coordinates_points = np.array([ [p.position[0],p.position[1]] for p in points])
	coordinates_ants = np.array([ [a.position[0],a.position[1]] for a in ants])


	paths = draw_all_paths(coordinates_points,feromone)
	
	ax.scatter(coordinates_points[:-2,0],coordinates_points[:-2,1],s=200,color='black',zorder=1)
	ax.scatter(coordinates_points[-2:,0],coordinates_points[-2:,1],s=200,color='orange',zorder=1)	# points A and B
	ax.scatter(coordinates_ants[:,0],coordinates_ants[:,1],s=2,color='red',zorder=1)				# all ants


	#begin main loop  ---- animation func is not implemented yet
	t = 0
	while t < 100:
		feromone = update_feromone(feromone,ants,points)
		paths = draw_all_paths(coordinates_points,feromone,paths)


		t += 1 
		break
	#ani = animation.FuncAnimation(fig, animate, np.arange(1, 200), blit=False, interval=10,repeat=False, init_func=init)
	plt.show()

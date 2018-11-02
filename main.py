import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


rho = 0.99  #evaporation coefficient
Q = 1		#constant quantity


class ant:
	''' ant object, has attributes of position'''
	def __init__(self,xy):
		self.position = np.array(xy)
		self.alive = True
		self.tabu = []
		self.last_path = [np.array([np.nan,np.nan]),np.array([np.nan,np.nan])]

	def make_move(self,feromone,waypoints):
		''' 
		takes feromone and a list of all distances to waypoints as arguments
		and then calls probability and decision function in order to make a move
		'''
		if not self.alive:						#check if alive
			return False
		d = self.calc_dist(waypoints)			#get distance dictionary
		self.tabu.append(self.position)			#append current position to tabu list
		probabilities = self.prob(feromone,d)	#get probabilities of each path (to waypoints)
		p = self.decision(probabilities)		#get decision
		if not self.alive:						#check if alive (decision() could have changed this state)
			return False
		self.position = p.position 	#make move (update position)
		self.last_path = [ self.tabu[-1], self.position ]	#set last path
		return True

	def calc_dist(self,waypoints):
		'''
		takes waypoints-list as argument
		returns a dictionary of all distances to the waypoints
		in the shape of:
		d = { 	<object1> : float,
				<object2> : float,
				....
			}
		where the keys of the dictionary are the waypoint objects
		(this makes it easier to find out which distance corresponds to which waypoint)
		'''
		d = {}
		for w in waypoints:
			d[w] = np.linalg.norm(self.position-w.position)
		return d

	def decision(self,probabilities):
		'''
		this function takes probabilities as argument but also needs to check if the decision 
		is not in the tabu list, else it should make another decision, and if no waypoint is left
		i.e. all are elements of the tabu list, then this function should call the kill_ant() function and return False
		'''

		#quinten
		
		choice = probabilities[0] #as a test, will always return the first possiblity
		return choice

	def prob(self,feromone,d):
		''' takes feromone strength and distances (list) to next point as arguments 
			returns the probabilities (list) of choosing these paths'''

		#this is not done yet, returns just one element for testing purposes
		p = [list(d.keys())[0]]
		return p

	def kill_ant(self):
		''' this function kills the ant, it should be called when it has visited all waypoints '''
		self.alive = False

class waypoint:
	num = 0
	''' class object of a waypoint, has attribute of position '''
	def __init__(self,xy):
		self.position = np.array(xy)
		self.id = self.num + 1
		waypoint.num = self.id

def delta(k,p,i,j):
	'''
	takes k-th ant, point list and indices i and j as arguments
	and returns value of 
	delta tau
	'''
	if ( (k.last_path[0] == p[i].position).all() and (k.last_path[1] == p[j].position).all() ) or ( (k.last_path[0] == p[j].position).all() and (k.last_path[1] == p[i].position).all() ):
		return Q/np.linalg.norm(k.last_path[0]-k.last_path[1])
	else:
		return 0


def update_feromone(f,ant_list,point_list):
	''' takes feromone matrix, ant list and point list as arguments and returns the new feromone matrix'''
	#linus (1. method ant quantity)
	#could be vectorized and split in half (since tau is traceless and symmetrical)


	f_new = np.copy(f)
	for i in range(len(f)):
		for j in range(len(f)):
			s = 0
			for k in ant_list:
				if i==j:
					continue
				s += f[i][j]*rho + delta(k,point_list,i,j)
			f_new[i][j] = s
	return f_new


######## initial
#for nora (travelling salesman)

def initial_condition(n=20,set_seed=True):
	'''
	takes number of waypoints and a boolean as arguments
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
	takes waypoints and feromone as arguments, optionally previously drawn paths
	this function then draws all paths according to the feromone level
	and returns a list of all matplotlib line-objects (so they can be deleted later on)
	'''
	x = []
	y = []
	if paths:
		for line in paths:
			#clear all previously drawn paths (saves memory)
			try:
				line.pop(0).remove()
			except:
				pass
	for i in range(len(p)):
		for j in range(len(p)):
			if i!=j:
				width = feromone[i][j]/10 #adjusts with according to feromone strength
				print(width)
				paths.append(ax.plot( [p[i][0], p[j][0]] ,[p[i][1], p[j][1]] ,color='blue',linewidth=width,zorder=-2))
				#print([p[i][0], p[j][0]], [p[i][1], p[j][1]])
				#paths[i][j].set_data( [p[i][0], p[j][0]], [p[i][1], p[j][1]] )
	return paths,

def run(t,feromone,paths):
	feromone = update_feromone(feromone,ants,points)
	paths = draw_all_paths(coordinates_points,feromone,paths)
	for a in ants:
		#let all ants make a move
		if not a.make_move(feromone,points):
			# if the ant is dead, add it to the list
			dead_ants.append(a)
	# after the loop above is finished we remove all dead ants from the ant list
	# we have to do this after finishing the loop in order to avoid list out of range error!
	for a in dead_ants:
		ants.remove(a)
	# if not ants:
	# 	#if no more ants are alive, break
	# 	return False
	return

def init():
    ax.set_xlim(0, 1)
    ax.set_ylim(0,1)
    return paths,


if __name__ == '__main__':
	fig, ax = plt.subplots()
	xdata, ydata = [], []
	points,ants,feromone = initial_condition()
	paths = [ [ '' for i in range(len(points))] for j in range(len(points)) ]

	for i in range(len(points)):
		for j in range(len(points)):
			ln, = ax.plot([], [], color='blue',linewidth=1,zorder=-2, animated=True)
			paths[i][j] = ln
	dead_ants = []

	#get list of coordinates for plotting purposes
	coordinates_points = np.array([ [p.position[0],p.position[1]] for p in points])
	coordinates_ants = np.array([ [a.position[0],a.position[1]] for a in ants])


	#paths = draw_all_paths(coordinates_points,feromone,paths)
	
	ax.scatter(coordinates_points[:-2,0],coordinates_points[:-2,1],s=200,color='black',zorder=1)
	ax.scatter(coordinates_points[-2:,0],coordinates_points[-2:,1],s=200,color='orange',zorder=1)	# points A and B
	ax.scatter(coordinates_ants[:,0],coordinates_ants[:,1],s=2,color='red',zorder=1)				# all ants


	#begin main loop  ---- animation func is not implemented yet

	ani = animation.FuncAnimation(fig, run, np.arange(1, 20), fargs=(feromone,paths),blit=False, interval=10,repeat=False, init_func=init)
	plt.show()

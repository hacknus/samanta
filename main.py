import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class ant:
	''' ant object'''
	def __init__(self,town,n,i):
		self.position = town
		self.i = i
		self.n = n
		self.tabu_mask = np.ones(n)
		self.last_path = [town,town]
		self.path_length_history = []
		self.tabu_mask[i] = 0		

	def make_move(self,paths,cities):
		''' 
		takes path matrix and city list as arguments and makes the move (according to probability):
			- setting new position
			- setting last path
			- setting last town to 0 in tabu list
		'''
		probabilities = self.prob(paths)					#get probabilities of each path (to cities)
		probabilities = probabilities[self.i]				#only look at the probabilities corresponding to this ant/town (from matrix to list)
		town = self.decision(probabilities,cities)			#get decision
		self.position = town 								#make move (update position)
		town_index = cities.index(self.last_path[1])
		#print(town_index,self.last_path[1].i)
		self.tabu_mask[town_index] = 0						#add last city to tabu mask
		self.last_path = [self.last_path[1],town]			#set last path
		self.path_length_history.append(paths[self.i][town_index].distance)

	def decision(self,probabilities,cities):
		'''
		this function takes a probability list as weights and chooses from the cities list, it returns the chosen town
		'''
		town = np.random.choice(cities,p=probabilities)
		return town

	def prob(self,paths):
		''' 
		takes path matrix as input and calculates probability matrix
		'''

		#vectorize?

		alpha = 1.
		beta = -5.
		n = self.n
		p = np.zeros((n,n))
		allowed = np.arange(n)[self.tabu_mask != 0]
		for i in range(n):
			for j in range(n):
				if i == j or self.tabu_mask[j] == 0:
					p[i][j] = 0
				else:
					p[i][j] = paths[i][j].feromone**alpha * paths[i][j].distance**beta / np.sum([paths[i][k].feromone**alpha * paths[i][k].distance**beta for k in allowed if i!= k])
		return p

	def __repr__(self):
		''' for debugging '''
		return str(self.position)		


class Path:
	''' Path object '''
	def __init__(self,i,j):
		self.i = i
		self.j = j
		self.feromone = 0.5 #initial feromone
		self.distance = 1

	def set_dist(self,a,b):
		''' sets path distance '''
		self.distance = np.linalg.norm(a.position-b.position)

	def __repr__(self):
		''' for debugging '''
		return str((self.distance,self.feromone))


class city:
	''' class object of a city, has attribute of position (for plotting) '''
	def __init__(self,xy,i):
		self.i = i
		self.position = np.array(xy)

	# def __repr__(self):
	# 	''' for debugging '''
	# 	return str(self.i)		


class Algorithm:
	''' total algorithm class, easy to import '''
	def __init__(self,n,rho=0.99,Q=1):
		self.n = n
		self.city_list = []		
		self.ant_list = []		
		self.paths = []
		self.fig = None
		self.ax = None	
		self.coordinates_cities = []
		self.coordinates_ants = []
		self.rho = rho  		#evaporation coefficient
		self.Q = Q				#constant quantity
		self.performance = []

	def delta(self,i,j):
		'''
		takes indices i and j as arguments
		and returns value of 
		delta tau (sum)
		'''
		s = 0
		for k in self.ant_list:
			if (k.last_path[0].i == i and k.last_path[1].i == j) or (k.last_path[0].i == j and k.last_path[1].i == i):
				s+= self.Q/self.paths[i][j].distance
		return s


	def update_feromone(self):
		'''
		updates the feromone attributes of the path matrix elements
		'''

		#vectorize?
		for i in range(len(self.paths)):
			for j in range(len(self.paths)):
				if i == j:
					self.paths[i][j].feromone = 0
				else:
					self.paths[i][j].feromone = self.paths[i][j].feromone*self.rho + self.delta(i,j)


	def initial_condition(self,set_seed=True):
		'''
		takes number of cities and a boolean as arguments
		if boolean is true, then the seed will be fixed such that on every run the 'random' values are the same
		this makes it easy to compare in the debug phase
		'''
		Apos = [0,0]	# city A
		Bpos = [1,1]	# city B

		n = self.n

		if set_seed:
			np.random.seed(0)			#to get each time the same random numbers

		for i in range(1,n-1):
			pos = np.random.rand(2)
			self.city_list.append(city(pos,i))
			self.ant_list.append(ant(self.city_list[i-1],n,i-1))
		A = city(Apos,0)
		B = city(Bpos,n-1)
		self.city_list.append(A)
		self.ant_list.append(ant(A,n,0))
		self.city_list.append(B)
		self.ant_list.append(ant(B,n,n-1))
		self.paths = [[ None for i in range(n)] for j in range(n) ]
		for i in range(n):
			for j in range(n):
				self.paths[i][j] = Path(i,j)
				self.paths[i][j].set_dist(self.city_list[i],self.city_list[j])

	def init_plot(self):
		'''
		initializes plot and plots the cities and ants as scatterplot
		'''
		self.fig, self.ax = plt.subplots()
		self.coordinates_cities = np.array([ [p.position[0],p.position[1]] for p in self.city_list])
		self.coordinates_ants = np.array([ [a.position.position[0],a.position.position[1]] for a in self.ant_list])
		self.ax.scatter(self.coordinates_cities[:-2,0],self.coordinates_cities[:-2,1],s=200,color='black',zorder=1)
		self.ax.scatter(self.coordinates_cities[-2:,0],self.coordinates_cities[-2:,1],s=200,color='orange',zorder=1)	# cities A and B
		self.ax.scatter(self.coordinates_ants[:,0],self.coordinates_ants[:,1],s=2,color='red',zorder=1)					# all ants


	def draw_all_paths(self,t):
		'''
		for animating, not really working yet
		'''
		p = self.coordinates_cities
		x = []
		y = []
		all_widths = np.zeros((n,n))
		for i in range(len(p)):
			for j in range(len(p)):
				if i!=j:
					width = np.log(self.paths[i][j].feromone)  #adjusts with according to feromone strength
					all_widths[i][j] = width
		for i in range(len(p)):
			for j in range(len(p)):
				if i!=j:
					width = all_widths[i][j]/np.sum(np.array(all_widths))/len(p)**2 #- 1.01*np.sum(np.array(all_widths))/len(p)**2 #adjusts with according to feromone strength
					color = self.map(width/np.sum(np.array(all_widths))/len(p)**2)
					#print(np.sum(np.array(all_widths))/len(p)**2,width)
					if t % 1 == 0:
						self.ax.plot( [p[i][0], p[j][0]] ,[p[i][1], p[j][1]] ,color='{}'.format('black'),linewidth=1.5*width,zorder=-2)
					#print([p[i][0], p[j][0]], [p[i][1], p[j][1]])
					#paths[i][j].set_data( [p[i][0], p[j][0]], [p[i][1], p[j][1]] )
		return

	def map(self,w):
		''' maps the width (feromone) to color '''
		return np.exp(-w)


	def animate(self,t):
		'''
		for animating, not really working yet
		'''
		self.update_feromone()
		self.draw_all_paths(t)
		for a in self.ant_list:
			#let all ants make a move
			a.make_move(self.paths,self.city_list)
			if np.count_nonzero(a.tabu_mask)==0:
				plt.savefig("ant.png")
				raise Exception("end of run")

	def init(self):
		'''
		for animating, not really working yet
		'''
		self.ax.set_xlim(0,1)
		self.ax.set_ylim(0,1)

	def run(self):
		'''
		for animating, not really working yet
		'''
		ani = animation.FuncAnimation(self.fig, self.animate, np.arange(0, self.n+1),blit=False, interval=10,repeat=False, init_func=self.init)
		plt.show()


	def shortest_path(self,counter):
		sp = []
		for a in self.ant_list:
			sp.append(sum(a.path_length_history))
		print(min(sp))
		self.performance.append(min(sp))

	def reload_ants(self):
		'''
		creates new ants (thus new empty tabu lists)
		but lets the feromones unchanged
		'''
		self.ant_list = []
		for i in range(0,self.n):
			self.ant_list.append(ant(self.city_list[i],n,i))

	def run_without_anim(self,counter):
		'''
		lets the ants make their moves until tabu list is filled, then saves the image and returns
		'''
		while True:
			self.update_feromone()
			self.draw_all_paths(counter)
			for a in self.ant_list:
				#let all ants make a move
				if np.count_nonzero(a.tabu_mask) == 0:
					plt.savefig("run{}.png".format(counter))
					plt.cla()
					return
				a.make_move(self.paths,self.city_list)



if __name__ == '__main__':

	n = 10
	cycle = Algorithm(n)
	cycle.initial_condition()
	cycle.init_plot()
	for i in range(200):
		cycle.run_without_anim(i)
		cycle.init_plot()
		cycle.shortest_path(i)
		cycle.reload_ants()
		print(i)
	plt.cla()
	plt.plot(range(200),cycle.performance)
	plt.show()


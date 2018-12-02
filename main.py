import numpy as np
import matplotlib.pyplot as plt

class ant:
	''' ant object'''
	def __init__(self,town,n,i):
		self.position = town
		self.i = i
		self.n = n
		self.tabu_mask = np.ones(n)
		self.last_path = [town,town]
		self.path_length_history = []
		self.history = [town]
		self.tabu_mask[i] = 0		

	def make_move(self,paths,cities):
		''' 
		takes path matrix and city list as arguments and makes the move (according to probability):
			- setting new position
			- setting last path
			- setting last town to 0 in tabu list
		'''
		probabilities = self.prob(paths)					#get probabilities of each path (to cities)
		town = self.decision(probabilities,cities)			#get decision
		self.position = town 								#make move (update position)
		new_town_index = cities.index(town)
		town_index = cities.index(self.last_path[1])
		self.tabu_mask[town_index] = 0						#add last city to tabu mask
		self.last_path = [self.last_path[1],town]			#set last path
		self.path_length_history.append(paths.distances[new_town_index][town_index])
		self.history.append(town)

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

		alpha = 10.
		beta = -5.
		n = self.n
		p = np.zeros(n)
		i = self.i
		self.tabu_mask[self.i] = 0

		p[self.tabu_mask != 0] = np.array(paths.feromones[i])[self.tabu_mask != 0]**alpha * np.array(paths.distances[i])[self.tabu_mask != 0]**beta / np.sum(	np.array(paths.feromones)[i][self.tabu_mask != 0]**alpha * np.array(paths.distances)[i][self.tabu_mask != 0]**beta )
		return p

	def __repr__(self):
		''' for debugging '''
		return str(self.position)		


class Paths:
	''' Path object '''
	def __init__(self,n,city_list):
		self.feromones = np.ones((n,n))*0.1 #initial feromone
		self.distances = np.ones((n,n))
		for i in range(n):
			for j in range(n):
				self.distances[i][j] = np.linalg.norm(city_list[i].position-city_list[j].position)


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
	def __init__(self,n,rho=0.99,Q=20):
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

	def update_feromone(self):
		'''
		updates the feromone attributes of the path matrix elements
		'''

		self.paths.feromones = self.paths.feromones*self.rho
		for k in self.ant_list:
			self.paths.feromones[k.last_path[0].i][k.last_path[1].i] += self.Q/self.paths.distances[k.last_path[0].i][k.last_path[1].i]
			self.paths.feromones[k.last_path[1].i][k.last_path[0].i] += self.Q/self.paths.distances[k.last_path[1].i][k.last_path[0].i]


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
		self.paths = Paths(n,self.city_list)

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

	def shortest_path(self,counter):
		sp = []
		for a in self.ant_list:
			sp.append(sum(a.path_length_history))
		print(min(sp))
		i = sp.index(min(sp))
		shortest = np.array([town.position for town in self.ant_list[i].history])
		plt.plot(shortest[:,0],shortest[:,1],color='red')
		self.performance.append(min(sp))
		return np.array([town.i for town in self.ant_list[i].history])

	def reload_ants(self,shortest):
		'''
		creates new ants (thus new empty tabu lists)
		'''
		self.ant_list = []
		for i in range(0,self.n):
			self.ant_list.append(ant(self.city_list[i],n,i))
		mask = np.zeros((self.n,self.n))
		for i in range(len(shortest)-1):
			mask[shortest[i]][shortest[i+1]] = 1
			mask[shortest[i+1]][shortest[i]] = 1
		self.paths.feromones[mask == 0] = 0.1

	def run(self,counter):
		'''
		lets the ants make their moves until tabu list is filled, then saves the image and returns
		'''
		while True:
			self.update_feromone()
			#self.draw_all_paths(counter)
			ant_count = 0
			for a in self.ant_list:
				#let all ants make a move
				if np.count_nonzero(a.tabu_mask) == 0:
					ant_count += 1
				else:
					a.make_move(self.paths,self.city_list)
				if ant_count == len(self.ant_list):
					l = self.shortest_path(counter)
					plt.savefig("run{}.png".format(counter))
					plt.cla()
					return l




if __name__ == '__main__':

	n = 20
	cycle = Algorithm(n)
	cycle.initial_condition()
	cycle.init_plot()
	for i in range(500):
		shortest = cycle.run(i)
		cycle.init_plot()
		cycle.reload_ants(shortest)
		print(i)
	plt.cla()
	plt.plot(range(len(cycle.performance)),cycle.performance)
	plt.savefig("performance.png")


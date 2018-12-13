import numpy as np
import matplotlib.pyplot as plt
from intersection import intersect

class ant:
	''' ant object'''
	def __init__(self,town,n,i):
		self.i = i
		self.n = n
		self.allowed = np.ones(self.n)
		self.path_length_history = []
		self.history = [town]
		self.allowed[self.i] = 0		

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
		self.i = cities.index(town)
		old_town_index = cities.index(self.history[-1])
		self.allowed[self.i] = 0							#add current city to tabu mask
		self.path_length_history.append(paths.distances[old_town_index][self.i])
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

		alpha = 1.
		beta = -5.

		p = np.zeros(self.n)
		z = np.array(paths.feromones[self.i])[self.allowed != 0]**alpha * np.array(paths.distances[self.i])[self.allowed != 0]**beta
		#print(z,self.allowed)
		p[self.allowed != 0] = z/np.sum(z)
		return p

	def __repr__(self):
		''' for debugging '''
		return str(self.position)		


class Paths:
	''' Path object '''
	def __init__(self,n,city_list):
		self.feromones = np.ones((n,n))*0.1 #initial feromone
		self.distances = np.zeros((n,n))
		for i in range(n):
			for j in range(n):
				self.distances[i][j] = np.linalg.norm(city_list[i].position-city_list[j].position)
				self.distances[j][i] = np.linalg.norm(city_list[i].position-city_list[j].position)
			if self.distances[i][j] != 0:
				self.feromones[i][j] = 1 / self.distances[i][j]
				self.feromones[j][i] = 1 / self.distances[j][i]
			else:
				self.feromones[i][j] = 0
				self.feromones[j][i] = 0

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
	def __init__(self,n,objects=False,rho=0.99,Q=100):
		self.n = n
		self.city_list = []
		self.ant_list = []
		self.objects = []
		self.o_boolean = objects
		self.paths = []
		self.fig = None
		self.ax = None	
		self.coordinates_cities = []
		self.coordinates_ants = []
		self.rho = rho			#evaporation coefficient
		self.Q = Q				#constant quantity
		self.performance = []
		if self.o_boolean:
			obst1 = [[0.1, 0.5], [0.3, 0.5]]
			obst2 = [[0.75, 0.8], [0.75, 0.5]]
			obst3 = [[0.25, 0.25], [1, 0.25]]
			self.objects.append(obst1)
			self.objects.append(obst2)
			self.objects.append(obst3)

	def update_feromone(self):
		'''
		updates the feromone attributes of the path matrix elements
		'''

		self.paths.feromones = self.paths.feromones*self.rho
		for k in self.ant_list:
			self.paths.feromones[k.history[-2].i][k.history[-1].i] += self.Q/self.paths.distances[k.history[-2].i][k.history[-1].i]
			self.paths.feromones[k.history[-1].i][k.history[-2].i] += self.Q/self.paths.distances[k.history[-2].i][k.history[-1].i]
		#self.paths.feromones[self.paths.feromones < 0.1] = 0.1		#prevent feromone levels from going below initial levels


	def initial_condition(self,set_seed=True):
		'''
		takes number of cities and a boolean as arguments
		if boolean is true, then the seed will be fixed such that on every run the 'random' values are the same
		this makes it easy to compare in the debug phase
		'''
		if set_seed:
			np.random.seed(0)			#to get each time the same random numbers

		for i in range(self.n):
			pos = np.random.rand(2)
			self.city_list.append(city(pos,i))
			self.ant_list.append(ant(self.city_list[i],self.n,i))
			self.ant_list.append(ant(self.city_list[i],self.n,i))

		self.paths = Paths(self.n,self.city_list)
		for i in range(self.n):
			for j in range(self.n):
				for k in range(len(self.objects)):
					if intersect(self.city_list[i].position,self.city_list[j].position,self.objects[k][0],self.objects[k][1]):
						self.paths.distances[i][j] = 10000
						self.paths.distances[j][i] = 10000
						self.paths.distances[j][i] = 10000

	def init_plot(self):
		'''
		initializes plot and plots the cities and ants as scatterplot
		'''
		#self.fig, self.ax = plt.subplots()
		self.coordinates_cities = np.array([ [p.position[0],p.position[1]] for p in self.city_list])
		plt.scatter(self.coordinates_cities[:,0],self.coordinates_cities[:,1],s=200,color='black',zorder=1)
		if self.o_boolean:
			for o in self.objects:
				plt.plot(np.array(o)[:,0],np.array(o)[:,1],color='red')

	def shortest_path(self,counter):
		sp = []
		for a in self.ant_list:
			sp.append(sum(a.path_length_history))
		print(min(sp))
		i = sp.index(min(sp))
		shortest = np.array([town.position for town in self.ant_list[i].history])
		if counter % 50 == 0 or counter == 55 or counter == 56:
			plt.title(str(min(sp)))
			plt.plot(shortest[:,0],shortest[:,1],color='blue')
		self.performance.append(min(sp))
		return np.array([town.i for town in self.ant_list[i].history])

	def reload_ants(self,shortest):
		'''
		creates new ants (thus new empty tabu lists)
		'''
		self.ant_list = []
		for i in range(self.n):
			self.ant_list.append(ant(self.city_list[i],self.n,i))

	def run(self,counter):
		'''
		lets the ants make their moves until tabu list is filled, then saves the image and returns
		'''
		while True:
			#self.draw_all_paths(counter)
			ant_count = 0
			for a in self.ant_list:
				#let all ants make a move
				if np.count_nonzero(a.allowed) == 0:
					ant_count += 1
					a.position = a.history[0]
					town_index = self.city_list.index(a.history[-1])
					a.i = town_index
					a.history.append(a.history[0])
					new_town_index = self.city_list.index(a.history[-1])
					a.path_length_history.append(self.paths.distances[town_index][new_town_index])
				else:
					a.make_move(self.paths,self.city_list)
			self.update_feromone()
			if ant_count == len(self.ant_list):
				l = self.shortest_path(counter)
				if counter % 50 == 0 or counter == 55 or counter == 56:
					plt.savefig("run{}.png".format(counter))
					plt.cla()
				return l





if __name__ == '__main__':

	n = 50
	cycle = Algorithm(n,objects=False)
	cycle.initial_condition()
	cycle.init_plot()
	for i in range(101):
		shortest = cycle.run(i)
		cycle.init_plot()
		cycle.reload_ants(shortest)
		print(i)
	plt.cla()
	plt.clf()
	plt.xlabel("runs")
	plt.ylabel("shortest path")
	plt.plot(range(len(cycle.performance)),cycle.performance)
	plt.savefig("performance.png")


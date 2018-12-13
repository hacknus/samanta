import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv


class ant:
	''' ant object'''
	def __init__(self,town,n,i):
		self.position = town
		self.i = i
		self.n = n
		self.allowed = np.ones(n)
		#self.last_path = [town]
		self.path_length_history = []
		self.history = [town]
		self.allowed[i] = 0
		self.dead = False
		self.finished = False

	def kill(self,finished=True):
		self.dead = True
		self.finished = finished
		print("killed ant, finished: ",finished)

	def make_move(self,paths,cities):
		''' 
		takes path matrix and city list as arguments and makes the move (according to probability):
			- setting new position
			- setting last path
			- setting last town to 0 in tabu list
		'''
		probabilities = self.prob(paths)					#get probabilities of each path (to cities)
		if type(probabilities) == bool:
			self.kill(False)
			return
		town = self.decision(probabilities,cities)			#get decision
		self.position = town 								#make move (update position)
		new_town_index = cities.index(town)
		town_index = cities.index(self.history[-1])
		self.i = new_town_index
		self.allowed[new_town_index] = 0						#add last city to tabu mask
		self.path_length_history.append(paths.distances[town_index][new_town_index])
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

		alpha = 1
		beta = -2.5
		gamma = -5
		n = self.n
		p = np.zeros(n)
		i = self.i
		allowed = paths.allowed_paths[i]
		self.allowed[i] = 0
		allowed = allowed*self.allowed
		if np.count_nonzero(allowed) == 0:
			return False
		#z = np.array(paths.feromones[i])[self.allowed != 0]**alpha * np.array(paths.distances[i])[self.allowed != 0]**beta
		if np.any(np.array(paths.target_distances)[allowed != 0] == 0):
			ptemp = np.zeros(n)
			ptemp[np.array(paths.target_distances) == 0] = 1
			p[allowed != 0] = ptemp[allowed != 0]
			return p
		else:
			z = np.array(paths.feromones[i])[allowed != 0]**alpha * np.array(paths.target_distances)[allowed != 0]**beta* np.array(paths.distances[i])[allowed != 0]**gamma		
			p[allowed != 0] = z/np.sum(z)
			return p


class Paths:
	''' Path object '''
	def __init__(self,n,city_list,indexB):
		self.feromones = np.ones((n,n))*0.1 #initial feromone
		self.distances = np.zeros((n,n))
		self.target_distances = np.zeros(n)
		self.allowed_paths = np.zeros((n,n))
		for i in range(n):
			for j in range(n):
				self.distances[i][j] = np.linalg.norm(city_list[i].position-city_list[j].position)
		for i in range(n):
				self.target_distances[i] = np.linalg.norm(city_list[indexB].position-city_list[i].position)


class city:
	''' class object of a city, has attribute of position (for plotting) '''
	def __init__(self,xy,i):
		self.i = i
		self.position = np.array(xy)

	def __repr__(self):
		''' for debugging '''
		return str(self.i)		


class Algorithm:
	''' total algorithm class, easy to import '''
	def __init__(self,rho=0.99,Q=100):
		self.city_list = []		
		self.ant_list = []		
		self.paths = []
		self.A = None
		self.B = None
		self.index_A = 298
		self.index_B = 23
		self.coordinates_cities = []
		self.coordinates_ants = []
		self.rho = rho  		#evaporation coefficient
		self.Q = Q				#constant quantity
		self.n = 0
		self.performance = []

	def update_feromone(self):
		'''
		updates the feromone attributes of the path matrix elements
		'''

		self.paths.feromones = self.paths.feromones*self.rho
		for k in self.ant_list:
			self.paths.feromones[k.history[-2].i][k.history[-1].i] += self.Q#/self.paths.distances[k.history[-2].i][k.history[-1].i]
			self.paths.feromones[k.history[-1].i][k.history[-2].i] += self.Q#/self.paths.distances[k.history[-1].i][k.history[-2].i]

	def initial_condition(self):
		'''
		takes number of cities and a boolean as arguments
		if boolean is true, then the seed will be fixed such that on every run the 'random' values are the same
		this makes it easy to compare in the debug phase
		'''

		data=[]
		with open("intersections.csv") as csvfile:
			readCSV = csv.reader(csvfile, delimiter=',')
			for out in readCSV:
				data.append([ int(out[0]),int(out[1])])

		self.n = len(data)
		n = self.n

		street_list = []
		street_indices = []
		with open("streets_allowed.csv") as csvfile:
			readCSV = csv.reader(csvfile, delimiter=',')
			for out in readCSV:
				street_list.append([[int(out[0]),int(out[1])],[int(out[2]),int(out[3])]])
				street_indices.append([data.index([int(out[0]),int(out[2])]),data.index([int(out[1]),int(out[3])])])
		street_list = np.array(street_list)
		data = np.array(data)


		for i in range(n):
			#pos = np.random.rand(2)
			pos = data[i]
			self.city_list.append(city(pos,i))

		self.index_A = 296
		self.index_B = 21
		self.A = self.city_list[self.index_A]
		self.B = self.city_list[self.index_B]

		for i in range(10):
			self.ant_list.append(ant(self.A,n,self.index_A))



		self.paths = Paths(n,self.city_list,self.index_B)
		self.paths.allowed_paths[np.array(street_indices)[:,0],np.array(street_indices)[:,1]] = 1
		self.paths.allowed_paths[np.array(street_indices)[:,1],np.array(street_indices)[:,0]] = 1
		# for i in range(n):
		# 	for j in range(n):
		# 		if i in np.array(street_indices)[:,0] and j in np.array(street_indices)[:,1]:
		# 			self.paths.allowed_paths[i][j] = 1



	def init_plot(self):
		'''
		initializes plot and plots the cities and ants as scatterplot
		'''
		#self.fig, self.ax = plt.subplots()
		self.coordinates_cities = np.array([ [p.position[0],p.position[1]] for p in self.city_list])
		self.coordinates_ants = np.array([ [a.position.position[0],a.position.position[1]] for a in self.ant_list])
		image = cv2.imread('hbirchel.png')
		plt.imshow(image)
		#plt.scatter(self.coordinates_cities[:,0],self.coordinates_cities[:,1],s=20,color='black',zorder=1)
		plt.scatter(self.coordinates_cities[self.index_A][0],self.coordinates_cities[self.index_A][1],s=20,color='orange',zorder=1)
		plt.scatter(self.coordinates_cities[self.index_B][0],self.coordinates_cities[self.index_B][1],s=20,color='orange',zorder=1)

	def shortest_path(self,counter):
		while True:
			sp = []
			if len(self.ant_list) == 0:
				return False
			for a in self.ant_list:
				sp.append(sum(a.path_length_history))
			i = sp.index(min(sp))
			if self.ant_list[i].finished:
				break
			else:
				self.ant_list.pop(i)
		print("shortest path: ",min(sp))
		shortest = np.array([town.position for town in self.ant_list[i].history])
		plt.title(str(min(sp)))
		plt.plot(shortest[:,0],shortest[:,1],color='red')
		self.performance.append(min(sp))
		return np.array([town.i for town in self.ant_list[i].history])

	def reload_ants(self,shortest):
		'''
		creates new ants (thus new empty tabu lists)
		'''
		self.ant_list = []
		for i in range(0,self.n):
			self.ant_list.append(ant(self.A,self.n,self.index_A))
		# mask = np.zeros((self.n,self.n))
		# for i in range(len(shortest)-1):
		# 	mask[shortest[i]][shortest[i+1]] = 1
		# 	mask[shortest[i+1]][shortest[i]] = 1
		# self.paths.feromones[mask == 0] = 0.1

	def run(self,counter):
		'''
		lets the ants make their moves until tabu list is filled, then saves the image and returns
		'''
		#c = 0
		while True:
			#self.draw_all_paths(counter)
			ant_count = 0
			for a in self.ant_list:
				#let all ants make a move
				if a.i == self.index_B and a.dead == False:
					a.kill()
				else:
					if not a.dead:
						a.make_move(self.paths,self.city_list)
				if a.dead == False:
					ant_count += 1
			# c+=1
			# if c > 500:
			# 	print("overflow")
			# 	ant_count = len(self.ant_list)
			if ant_count == 0:
				self.update_feromone()
				l = self.shortest_path(counter)
				if type(l) == bool:
					print("NO ANT HAS REACHED DESTINATION")
				if counter % 1 == 0:
					plt.savefig("run{}.png".format(counter),dpi=300)
					plt.cla()
				return l

			self.update_feromone()





if __name__ == '__main__':

	cycle = Algorithm()
	cycle.initial_condition()
	cycle.init_plot()
	for i in range(50):
		shortest = cycle.run(i)
		cycle.init_plot()
		cycle.reload_ants(shortest)
		print(i)
	plt.cla()
	plt.plot(range(len(cycle.performance)),cycle.performance)
	plt.savefig("performance.png")


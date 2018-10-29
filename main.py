import numpy as np
import matplotlib.pyplot as plt


rho = 0.99  #evaporation coefficient



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



def update_feromone(f,ant_list):
	''' takes feromone matrix as input and'''
	#linus (1. method ant quantity)
	#could be vectorized


	f_new = np.copy(f)
	for i in range(len(f)):
		for j in range(len(f)):
			pass

def decision(probabilities):
	#quinten
	choice = 0
	return choice

def prob(feromone,d):
	''' takes feromone strength and distance to next point as input 
		returns the probability of choosing this path'''
	p = 1
	return p


def main():
	t = 0
	while t < 100:
		break

######## initial
#for nora (travelling salesman)


np.random.seed(0)			#to get each time the same random numbers

points = []
ants = []
for i in range(10):
	pos = np.random.rand(2)
	points.append(waypoint(pos))
	ants.append(ant(pos))

########

n = len(points)
feromone = np.ones((n,n)) #initialize feromone at t0





if __name__ == '__main__':
	main()

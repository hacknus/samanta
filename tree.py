import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


class node:

	def __init__(self,A,bound,d,ax=None,num=8):
		self.A = A
		self.bound = bound
		self.d = d
		self.num = num
		self.ax = ax
		self.mass_tot = None
		self.com = None
		self.quad = None
		self.particles = []
		self.isLeaf = False

	def plot_init(self):
		fig = plt.figure()
		self.ax = fig.add_subplot(111)
		x = [ a[0] for a in self.A]
		y = [ a[1] for a in self.A]
		self.ax.scatter(x,y,s=2,c='red')
		return self.ax
 
	def buildTree(self,a,b):
		if len(self.A[a:b]) <= self.num:
			#is leaf
			self.particles = self.A[a:b]
			self.isLeaf = True
			return
		if self.ax:
			self.drawCuts()
		v = self.bound[0][self.d] + (self.bound[1][self.d]-self.bound[0][self.d])/2.
		i = self.Partition(a,b,v)
		upper_bound = np.copy(self.bound)
		lower_bound = np.copy(self.bound)
		upper_bound[0][self.d] = v
		lower_bound[1][self.d] = v
		dimension = self.d + 1
		if dimension >= 2:
			dimension = 0
		self.lower = node(self.A,lower_bound,dimension,self.ax,self.num)
		self.upper = node(self.A,upper_bound,dimension,self.ax,self.num)
		if i >= a:
			self.lower.buildTree(a,i)
		if i < b:
			self.upper.buildTree(i,b)


	def Partition(self,a,b,v):
		i = a
		j = b-1
		switch_j = switch_i = False
		while True:
			if self.A[i][self.d] <= v:
				i+=1
			else:
				switch_i = True
			if self.A[j][self.d] > v:
				j-=1
			else:
				switch_j = True
			if switch_i and switch_j:
				temp = np.copy(self.A)[i]
				self.A[i] = self.A[j]
				self.A[j] = temp
				switch_j = switch_i = False
			if i > j:
				return i


	def drawCuts(self):
		v = self.bound[0][self.d] + (self.bound[1][self.d]-self.bound[0][self.d])/2.
		if self.d == 0:
			x1 = [ v,v ]
			y1 = [ self.bound[0][1],self.bound[0][1] ]
			x2 = [ v,v ]
			y2 = [ self.bound[1][1],self.bound[1][1] ]
			x3 = [ v,v ]
			y3 = [ self.bound[1][1],self.bound[0][1] ]
			x4 = [ v,v ]
			y4 = [ self.bound[0][1],self.bound[1][1] ]
		elif self.d == 1:
			x1 = [ self.bound[0][0],self.bound[0][0] ]
			y1 = [ v,v ]
			x2 = [ self.bound[0][0],self.bound[1][0] ]
			y2 = [ v,v ]
			x3 = [ self.bound[1][0],self.bound[1][0] ]
			y3 = [ v,v ]
			x4 = [ self.bound[1][0],self.bound[0][0] ]
			y4 = [ v,v ]

		self.ax.plot(x1,y1,color='blue',linewidth=0.5)
		self.ax.plot(x2,y2,color='blue',linewidth=0.5)
		self.ax.plot(x3,y3,color='blue',linewidth=0.5)
		self.ax.plot(x4,y4,color='blue',linewidth=0.5)

def checkLeaf(r):
	if r.isLeaf:
		print(r.com)
		return
	try:
		checkLeaf(r.lower)
	except AttributeError:
		pass
	try:
		checkLeaf(r.upper)
	except AttributeError:
		pass

def read_image(path="object.png"):
	img = mpimg.imread(path)
	img_inv = np.zeros((len(img),len(img)))
	img_inv[img==0] = 1
	x = []
	y = []
	for i in range(len(img_inv)):
		for j in range(len(img_inv)):
			if img_inv[i][j] == 1:
				x.append([i/len(img_inv),j/len(img_inv)])
	return x,img_inv



if __name__ == "__main__":
	n = 100
	dim = 7
	A = np.random.rand(n,dim)
	A,img_inv =	read_image()
	coord = np.array([[0.,0.],[1.,1.]])
	root = node(A,coord,d=0,num=10)
	ax = root.plot_init()
	#plt.scatter(A[0],A[1],cmap='gray')
	root.buildTree(a=0,b=n)
	plt.show()
import numpy as np

def label(image):
	""" 
    Args:
        image: binary inverted test image
    Returns:
        np image array with character in test image replaced by its label
    """

	width = len(image[0])
	height = len(image)

	labelled_image = np.zeros((height, width))
	uf = UnionFind() 
	current_label = 1

	# record label equivalences and label image
	for y, row in enumerate(image):
		for x, pixel in enumerate(row):
			
			if pixel == 0:
				pass
			else: 
				labels = neighbouring_labels(labelled_image, x, y)

				if not labels:
					labelled_image[y,x] = current_label
					uf.MakeSet(current_label) 
					current_label = current_label + 1 
				else:
					smallest_label = min(labels)
					labelled_image[y,x] = smallest_label

					if len(labels) > 1: 
						for label in labels:
							uf.Union(uf.GetNode(smallest_label), uf.GetNode(label))

	final_labels = {}
	new_label_number = 1

	for y, row in enumerate(labelled_image):
		for x, pixel_value in enumerate(row):
			
			if pixel_value > 0: 
				new_label = uf.Find(uf.GetNode(pixel_value)).value 
				labelled_image[y,x] = new_label

				if new_label not in final_labels:
					final_labels[new_label] = new_label_number
					new_label_number = new_label_number + 1

	for y, row in enumerate(labelled_image):
		for x, pixel_value in enumerate(row):
			
			if pixel_value > 0: # Foreground pixel
				labelled_image[y,x] = final_labels[pixel_value]

	return labelled_image



# Private functions ############################################################################
def neighbouring_labels(image, x, y):
	"""
	Args:
        image: binary inverted test image
		x, y: pixel location coordinates
    Returns:
        list of labels on left, top, top-left & top-right  
	"""

	labels = set()
	
	# Left neighbour
	if x > 0:
		left = image[y,x-1]
		if left > 0:
			labels.add(left)

	# Top neighbour
	if y > 0:
		top = image[y-1,x]
		if top > 0:
			labels.add(top)

	# Top Left neighbour
	if x > 0 and y > 0:
		top_left = image[y-1,x-1]
		if top_left > 0:
			labels.add(top_left)

	# Top Right neighbour
	if y > 0 and x < len(image[y]) - 1:
		top_right = image[y-1,x+1]
		if top_right > 0:
			labels.add(top_right)

	return labels


class UnionFind:

	def __init__(self):
		self.__nodes_by_value = {}

	def MakeSet(self, value):
		""" 
    	Args:
        	value: Current label value 
    	Returns:
			a new set containing single node
    	"""
		
		if self.GetNode(value):
			return self.GetNode(value)

		node = Node(value)

		self.__nodes_by_value[value] = node
		
		return node


	def Find(self, x):
		""" 
    	Args:
        	x: the node for which the parent needs to be searched 
    	Returns:
			the parent node
    	"""

		if x.parent  != x:  
			x.parent = self.Find(x.parent) 
		return x.parent


	def Union(self, x, y):
		""" 
    	Args:
        	x, y: the nodes whose sets need to be merged depending on the rank
    	"""

		if x == y:
			return

		x_root = self.Find(x)
		y_root = self.Find(y)

		if x_root == y_root:
			return

		if x_root.rank > y_root.rank: 
			y_root.parent = x_root

		elif x_root.rank < y_root.rank: 
			x_root.parent = y_root

		else: 
			x_root.parent = y_root
			y_root.rank = y_root.rank + 1

	
	def GetNode(self, value): 
		
		if value in self.__nodes_by_value:
			return self.__nodes_by_value[value]
		else:
			return False


class Node(object):
	def __init__(self, value):
		self.value = value
		self.parent = self 
		self.rank = 0 

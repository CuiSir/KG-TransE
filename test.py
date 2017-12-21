# -*- coding: UTF-8 -*-

class Test(object):
	"""docstring for Test"""
	def __init__(self, entityList, entityVectorList, relationList, relationVectorList, tripleListTrain, tripleListTest, label = "head", isFit = False):

		self.entityList = {}
		self.relationList = {}
		#zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
		for name, vec in zip(entityList, entityVectorList):
			self.entityList[name] = vec

		for name, vec in zip(relationList, relationVectorList):
			self.relationList[name] = vec
		self.tripleListTrain = tripleListTrain
		self.tripleListTest = tripleListTest
		self.rank = []
		self.label = label
		self.isFit = isFit

	def writeRank(self, dir):
		pass


def openD(dir, sp="\t"):
	#三元组为triple=(head, tail, relation)
	num = 0
	list = []
	with open(dir) as file:
		lines = file.readlines()
		for line in lines:
			triple = line.strip().split(sp)
			if(len(triple) < 3):
				continue
			list.append(tuple(triple))
			num += 1
	print(num)
	return num, list



if __name__ == '__main__':
	dirTrain = "C:\\Users\\cuicui\\Desktop\\transE\\data\\FB15k\\train.txt"
	tripleNumTrain, tripleListTrain = openD(dirTrain)
	dirTest = "C:\\Users\\cuicui\\Desktop\\transE\\data\\FB15k\\test.txt"
	tripleNumTest, tripleListTest = openD(dirTest)

# -*- coding: UTF-8 -*-
from numpy import *
from random import uniform
class TransE:
	"""docstring for TransE"""


	#初始化边界值，SGD学习率，降维后的维度，以及距离函数使用的范数
	def __init__(self, entityList, relationList, tripleList, margin = 1, learningRate = 0.00001, dim = 10, L1 = True):
		self.margin = margin
		self.learningRate = learningRate
		self.dim = dim
		self.L1 = L1
		#entityList为entity的list，初始化后，类型为字典，key为entity的id，value为对应的向量
		self.entityList = entityList
		self.relationList = relationList
		self.tripleList = tripleList
		self.loss = 0


	#初始化实体和关系向量
	def initialize(self):
		'''
		初始化向量
		'''
		entityVectorList = {}
		relationVectorList = {}

		#初始化实体向量
		for entity in self.entityList:
			n = 0
			entityVector = []
			while  n < self.dim:
				ram = init(self.dim) #初始化的范围
				entityVector.append(ram)
				n += 1
			entityVector = norm(entityVector) #实体向量归一化
			entityVectorList[entity] = entityVector
		print("entityVector初始化完成，数量是%d"%len(entityVectorList)).decode('UTF-8').encode('cp936')

		#初始化关系向量
		for relation in self.relationList:
			n = 0
			relationVector = []
			while n < self.dim:
				ram = init(self.dim) #初始化的范围
				relationVector.append(ram)
				n += 1
			relationVector = norm(relationVector) #关系向量归一化
			relationList[relation] = relationVector
		print("relationVector初始化完成，数量是%d"%len(relationVectorList)).decode('UTF-8').encode('cp936')

		self.entityList = entityVectorList
		self.relationList = relationVectorList


	#transE训练
	def transE(self, cI = 20):
		print("训练开始").decode('UTF-8').encode('cp936')
		for cycleIndex in range(cI):
			pass


	#random.sample()list中，随机的截取指定长度size的片断
	def getSample(self, size):
		return sample(self.tripleList, size)

	#由(h,r,t)生成(h',r,t)或者(h,r,t')
	def getCorruptedTriplet(self, triplet):
		i = uniform(-1,1)
		if i < 0: #小于0，打破三元组的头实体
			while True:
				entityTemp = sample(self.entityList.keys(),1)[0]
				if entityTemp != triplet[0]:
					break
			corruptedTriplet = (entityTemp, triplet[1], triplet[2])

		else: #大于等于0，打坏三元组的尾实体
			while True:
				entityTemp = sample(self.entityList.keys(),1)[0]
				'''
				这里换的是triplet[2]还是triplet[1]
				'''
				if entityTemp != triplet[2]:
					break
			corruptedTriplet = (triplet[0], triplet[1], entityTemp)

		return corruptedTriplet





#对关系进行归一化处理，使用random模块中的uniform函数生成范围中的随机浮点数
def init(dim):
	return uniform(-6/(dim**0.5), 6/(dim**0.5))


#计算L1范数下的d，使用到numpy库中的对应元素的计算，以及fabs计算每个元素的绝对值
def distanceL1(h,t,r):
	s = h + r - t
	sum = fabs(s).sum()
	return sum

#计算L2范数下的d
def distanceL2(h,t,r):
	s = h + r -t
	sum = (s*s).sum()
	return sum

def norm(list):
	'''
	将向量归一化，输出向量每一项除以向量的模（平方和开方）
	'''
	var = linalg.norm(list) #使用numpy中的norm函数，默认为L2范数，对应向量的模
	while i < len(list):
		list[i] = list[i]/var
		i += 1
	return array(list) #返回numpy向量，可对每个元素进行加减
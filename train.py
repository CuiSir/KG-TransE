# -*- coding: UTF-8 -*-
from numpy import *
from random import uniform, sample
from copy import deepcopy
class TransE:
	"""docstring for TransE"""


	#初始化边界值，SGD学习率（步长），降维后的维度，以及距离函数使用的范数
	def __init__(self, entityList, relationList, tripleList, margin = 1, learningRate = 0.01, dim = 10, L1 = True):
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
			relationVectorList[relation] = relationVector
		print("relationVector初始化完成，数量是%d"%len(relationVectorList)).decode('UTF-8').encode('cp936')

		self.entityList = entityVectorList
		self.relationList = relationVectorList


	#transE训练，cI为最大迭代次数
	def transE(self, cI = 20):
		print("训练开始").decode('UTF-8').encode('cp936')
		for cycleIndex in range(cI):
			Sbatch = self.getSample(150) 
			Tbatch = [] #三元组对（原三元组，破坏后的三元组）列表：[{(h,r,t),(h',r,t')}]

			for sbatch in Sbatch:
				tripletWithCorruptedTriplet = (sbatch,self.getCorruptedTriplet(sbatch))
				if (tripletWithCorruptedTriplet not in Tbatch):
					Tbatch.append(tripletWithCorruptedTriplet)

			self.update(Tbatch)
			if cycleIndex % 100 == 0:
				print("第%d次循环"%cycleIndex).decode('UTF-8').encode('cp936')
				print(self.loss)
				self.writeRelationVector("C:\\Users\\cuicui\\Desktop\\transE\\result\\relationVector.txt")
				self.writeEntityVector("C:\\Users\\cuicui\\Desktop\\transE\\result\\entityVector.txt")
				self.loss = 0

	#梯度下降，更新向量
	def update(self, Tbatch):
		copyEntityList = deepcopy(self.entityList)
		copyRelationList = deepcopy(self.relationList)

		for tripletWithCorruptedTriplet in Tbatch: #Tbatch中是原三元组和破坏后的三元组
			headEntityVector = copyEntityList[tripletWithCorruptedTriplet[0][0]]
			tailEntityVector = copyEntityList[tripletWithCorruptedTriplet[0][1]]
			relationVector = copyRelationList[tripletWithCorruptedTriplet[0][2]]

			headEntityVectorWithCorruptedTriplet = copyEntityList[tripletWithCorruptedTriplet[1][0]]
			tailEntityVectorWithCorruptedTriplet = copyEntityList[tripletWithCorruptedTriplet[1][1]]


			headEntityVectorBeforeBatch = self.entityList[tripletWithCorruptedTriplet[0][0]]
			tailEntityVectorBeforeBatch = self.entityList[tripletWithCorruptedTriplet[0][1]]
			relationVectorBeforeBatch = self.relationList[tripletWithCorruptedTriplet[0][2]]

			headEntityVectorWithCorruptedTripletBeforeBatch = self.entityList[tripletWithCorruptedTriplet[1][0]]
			tailEntityVectorWithCorruptedTripletBeforeBatch = self.entityList[tripletWithCorruptedTriplet[1][1]]

			#使用L1范数
			if self.L1:
				distTriplet = distanceL1(headEntityVectorBeforeBatch,tailEntityVectorBeforeBatch,relationVectorBeforeBatch)
				distCorruptedTriplet = distanceL1(headEntityVectorWithCorruptedTripletBeforeBatch,tailEntityVectorWithCorruptedTripletBeforeBatch,relationVectorBeforeBatch)

			else: #使用L2范数
				distTriplet = distanceL2(headEntityVectorBeforeBatch,tailEntityVectorBeforeBatch,relationVectorBeforeBatch)
				distCorruptedTriplet = distanceL2(headEntityVectorWithCorruptedTripletBeforeBatch,tailEntityVectorWithCorruptedTripletBeforeBatch,relationVectorBeforeBatch)


			eg = self.margin + distTriplet - distCorruptedTriplet
			if eg > 0: #[function]+ 是一个取正值的函数，合页损失函数，大于零取原值，小于零则为零
				self.loss += eg
				if self.L1:
					tempPositive = 2 * self.learningRate * (tailEntityVectorBeforeBatch - headEntityVectorBeforeBatch - relationVectorBeforeBatch)
					tempNegative = 2 * self.learningRate * (tailEntityVectorWithCorruptedTripletBeforeBatch - headEntityVectorWithCorruptedTripletBeforeBatch - relationVectorBeforeBatch)
					tempPositiveL1 = []
					tempNegativeL1 = []

					for i in range(self.dim):#如果没有0元素，可以使用numpy的array除以绝对值array，a/|a|
						if tempPositive[i] >= 0:
							tempPositiveL1.append(1)
						else:
							tempPositiveL1.append(-1)

						if tempNegative[i] >= 0:
							tempNegativeL1.append(1)
						else:
							tempNegativeL1.append(-1)

					tempPositive = array(tempPositiveL1)
					tempNegative = array(tempNegativeL1)

				else: #L2
					tempPositive = 2 * self.learningRate * (tailEntityVectorBeforeBatch - headEntityVectorBeforeBatch - relationVectorBeforeBatch)
					tempNegative = 2 * self.learningRate * (tailEntityVectorWithCorruptedTripletBeforeBatch - headEntityVectorWithCorruptedTripletBeforeBatch - relationVectorBeforeBatch)

				#缩短正确元组中t和h+r的距离
				headEntityVector = headEntityVector + tempPositive
				tailEntityVector = tailEntityVector - tempPositive
				relationVector = relationVector + tempPositive - tempNegative
				
				#增加错误元祖中t和h+r的距离
				headEntityVectorWithCorruptedTriplet = headEntityVectorWithCorruptedTriplet - tempNegative
				tailEntityVectorWithCorruptedTriplet = tailEntityVectorWithCorruptedTriplet + tempNegative

				#归一化调整过后的向量
				copyEntityList[tripletWithCorruptedTriplet[0][0]] = norm(headEntityVector)
				copyEntityList[tripletWithCorruptedTriplet[0][1]] = norm(tailEntityVector)
				copyRelationList[tripletWithCorruptedTriplet[0][2]] = norm(relationVector)
				copyEntityList[tripletWithCorruptedTriplet[1][0]] = norm(headEntityVectorWithCorruptedTriplet)
				copyEntityList[tripletWithCorruptedTriplet[1][1]] = norm(tailEntityVectorWithCorruptedTriplet)
		
		self.entityList = copyEntityList
		self.relationList = copyRelationList


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
				数据中的顺序为 e1,e2,r
				'''
				if entityTemp != triplet[1]:
					break
			corruptedTriplet = (triplet[0], entityTemp, triplet[2])

		return corruptedTriplet


	#写入实体
	def writeEntityVector(self,dir):
		print("写入实体").decode('UTF-8').encode('cp936')
		entityVectorFile = open(dir,'w')
		for entity in self.entityList.keys():
			entityVectorFile.write(entity+"\t")
			entityVectorFile.write(str(self.entityList[entity].tolist())) #numpy的tolist，转为列表
			entityVectorFile.write("\n")
		entityVectorFile.close()


	#写入关系
	def writeRelationVector(self,dir):
		print("写入关系").decode('UTF-8').encode('cp936')
		relationVectorFile = open(dir,'w')
		for relation  in self.relationList.keys():
			relationVectorFile.write(relation + "\t")
			relationVectorFile.write(str(self.relationList[relation].tolist()))
			relationVectorFile.write("\n")

		relationVectorFile.close()


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
	i = 0
	while i < len(list):
		list[i] = list[i]/var
		i += 1
	return array(list) #返回numpy向量，可对每个元素进行加减


def openDetailsAndId(dir,sp="\t"):
	idNum = 0
	list = []
	with open(dir) as file:
		lines = file.readlines()
		for line in lines:
			DetailsAndId = line.strip().split(sp)
			list.append(DetailsAndId[0])
			idNum += 1

	return idNum, list


def openTrain(dir, sp="\t"):
	num = 0
	list = []
	with open(dir) as file:
		lines = file.readlines()
		for line in lines:
			triplet = line.strip().split(sp) #strip()默认移除空格
			if(len(triplet) < 3): #不完整的三元组
				continue
			list.append(tuple(triplet))
			num += 1
	return num, list

if __name__ == '__main__':
	dirEntity = "C:\\Users\\cuicui\\Desktop\\transE\\data\\FB15k\\entity2id.txt"
	entityIdNum, entityList = openDetailsAndId(dirEntity)
	dirRelation = "C:\\Users\\cuicui\\Desktop\\transE\\data\\FB15k\\relation2id.txt"
	relationIdNum, relationList = openDetailsAndId(dirRelation)
	dirTrain = "C:\\Users\\cuicui\\Desktop\\transE\\data\\FB15k\\train.txt"
	tripleNum, tripleList = openTrain(dirTrain)

	print("开始TransE").decode('UTF-8').encode('cp936')
	transE = TransE(entityList,relationList,tripleList,margin = 1, dim = 50)
	print("TransE初始化").decode('UTF-8').encode('cp936')
	transE.initialize()
	transE.transE(1000)
	transE.writeRelationVector("C:\\Users\\cuicui\\Desktop\\transE\\result\\relationVector.txt")
	transE.writeEntityVector("C:\\Users\\cuicui\\Desktop\\transE\\result\\entityVector.txt")

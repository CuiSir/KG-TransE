# -*- coding: UTF-8 -*-
from numpy import *
import operator

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

	def getRank(self):
		cou = 0
		for triplet in self.tripleListTest:
			rankList = {}
			for entityTemp in self.entityList.keys():
				if self.label == "head":
					corruptedTriplet = (entityTemp, triplet[1], triplet[2])
					if self.isFit and (corruptedTriplet in self.tripleListTrain):
						continue
					rankList[entityTemp] = distance(self.entityList[entityTemp],self.entityList[triplet[1]],self.relationList[triplet[2]])
				else:
					corruptedTriplet = (triplet[0], entityTemp, triplet[2])
					if self.isFit and (corruptedTriplet in self.tripleListTrain):
						continue
					rankList[entityTemp] = distance(self.entityList[triplet[0]],self.entityList[entityTemp],self.relationList[triplet[2]])
			#key为函数，指定取待排序元素的哪一项进行排序
			#operator.itemgetter(i)获取对象的第i个域的值		
			nameRank = sorted(rankList.items(), key = operator.itemgetter(1))
			if self.label == 'head':
				numTri = 0
			else:
				numTri = 1

			x = 1
			for i in nameRank:
				if i[0] == triplet[numTri]:
					break
				x += 1

			self.rank.append((triplet, triplet[numTri], nameRank[0][0], x))
			#print(x)
			cou += 1
			if cou % 10000 == 0:
				print("Entity 第%d次循环"%cou).decode('UTF-8').encode('cp936')

	def getMeanRank(self):
		num = 0
		for r in self.rank:
			num += r[3]
		return num/len(self.rank)


	def getRelationRank(self):
		cou = 0
		self.rank = []
		for triplet in self.tripleListTest:
			rankList = {}
			for relationTemp in self.relationList.keys():
				corruptedTriplet = (triplet[0], triplet[1], relationTemp)
				if self.isFit and (corruptedTriplet in self.tripleListTrain):
					continue
				rankList[relationTemp] = distance(self.entityList[triplet[0]],self.entityList[triplet[1]],self.relationList[relationTemp])
			nameRank = sorted(rankList.items(), key = operator.itemgetter(1))
			x = 1
			for i in nameRank:
				if i[0] == triplet[2]:
					break
				x += 1
			self.rank.append((triplet, triplet[2], nameRank[0][0], x))
			#print(x)
			cou += 1
			if cou % 10000 == 0:
				print("Relation 第%d次循环"%cou).decode('UTF-8').encode('cp936')


	def writeRank(self,dir):
		print("写入").decode('UTF-8').encode('cp936')
		file = open(dir,'w')
		for r in self.rank:
			file.write(str(r[0]) + "\t")
			file.write(str(r[1]) + "\t")
			file.write(str(r[2]) + "\t")
			file.write(str(r[3]) + "\t")
		file.close()



def distance(h, t, r):
	h = array(h)
	t = array(t)
	r = array(r)
	s = h + r - t
	return linalg.norm(s) #返回s的L2范数的值

def openTrain(dir, sp="\t"):
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

def loadData(str):
	fr = open(str)
	sArr = [line.strip().split("\t") for line in fr.readlines()]
	datArr = [[float(s) for s in line[1][1:-1].split(", ")] for line in sArr]
	nameArr = [line[0] for line in sArr]
	return datArr, nameArr

if __name__ == '__main__':
	#C:\\Users\\cuicui\\Desktop\\transE\\data\\FB15k\\train.txt

	print("载入数据").decode('UTF-8').encode('cp936')
	dirTrain = "C:\\Users\\cuicui\\Desktop\\transE\\data\\FB15k\\train.txt"
	tripleNumTrain, tripleListTrain = openTrain(dirTrain)
	
	dirTest = "C:\\Users\\cuicui\\Desktop\\transE\\data\\FB15k\\test.txt"
	tripleNumTest, tripleListTest = openTrain(dirTest)
	
	dirEntityVector = "C:\\Users\\cuicui\\Desktop\\transE\\result\\entityVector.txt"
	entityVectorList, entityList = loadData(dirEntityVector)

	dirRelationVector = "C:\\Users\\cuicui\\Desktop\\transE\\result\\relationVector.txt"
	relationVectorList, relationList = loadData(dirRelationVector)

	print("开始测试").decode('UTF-8').encode('cp936')

	testHeadRaw = Test(entityList, entityVectorList, relationList, relationVectorList, tripleListTrain, tripleListTest)
	testHeadRaw.getRank()
	print("Head MeanRank: ")
	print(testHeadRaw.getMeanRank())
	testHeadRaw.writeRank("C:\\Users\\cuicui\\Desktop\\transE\\result\\testHeadRaw.txt")

	testHeadRaw.getRelationRank()
	print("Relation MeanRank: ")
	print(testHeadRaw.getMeanRank())
	testHeadRaw.writeRank("C:\\Users\\cuicui\\Desktop\\transE\\result\\testRelationRaw.txt")


	testTailRaw = Test(entityList, entityVectorList, relationList, relationVectorList, tripleListTrain, tripleListTest, label = "tail")
	testTailRaw.getRank()
	print("Tail MeanRank: ")
	print(testTailRaw.getMeanRank())
	testTailRaw.writeRank("C:\\Users\\cuicui\\Desktop\\transE\\result\\testTailRaw.txt")


	testHeadFit = Test(entityList, entityVectorList, relationList, relationVectorList, tripleListTrain, tripleListTest, isFit = True)
	testHeadFit.getRank()
	print("Head Fit MeanRank: ")
	print(testHeadFit.getMeanRank())
	testHeadFit.writeRank("C:\\Users\\cuicui\\Desktop\\transE\\result\\testHeadFit.txt")

	testHeadFit.getRelationRank()
	print("Relation Fit MeanRank: ")
	print(testHeadFit.getMeanRank)
	testHeadFit.writeRank("C:\\Users\\cuicui\\Desktop\\transE\\result\\testRelationFit.txt")

	testTailFit = Test(entityList, entityVectorList, relationList, relationVectorList, tripleListTrain, tripleListTest, isFit = True, label = "tail")
	testTailFit.getRank()
	print("Tail Fit MeanRank: ")
	print(tesTailFit.getMeanRank())
	testHeadFit.writeRank("C:\\Users\\cuicui\\Desktop\\transE\\result\\testTailFit.txt")
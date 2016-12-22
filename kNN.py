from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0] #shape 取矩阵长度，有多少行
    diffMat = tile(inX, (dataSetSize,1)) - dataSet #tile:重复列表，构建一个和dataset类似的矩阵。 并求减法
    sqDiffMat = diffMat**2 #差值做平方操作，数列里面每个数字都平方
    sqDistances = sqDiffMat.sum(axis=1) # 平方后，每一行加总
    distances = sqDistances**0.5 #开方
    sortedDistIndicies = distances.argsort() #返回从小到大的索引值，最小的是第几位，依次排开
    classCount={}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]] #计数
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    #sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True) #python 3.0没有iteritems用法
    sortedClassCount = sorted(classCount.items(), key = lambda e:e[1], reverse = True) #排序，输出排序后的列表
    return sortedClassCount[0][0]

def file2matrix(filename): #文件转化为向量矩阵输出和结果列表
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    returnMat = zeros((numberOfLines, 3)) #生成一个n行的，以0填充的矩阵
    classLabelVector = []
    index = 0
    for line in arrayOfLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

def autoNorm(dataSet): #标准化函数，将数值取0~1
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet, ranges, minVals

def datingClassTest(hoRatio=0.1, k=3): #hoRatio 测试样本占比 k kNN算法参数，取k个临近的向量
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m],k)
        print ("the classifier came back with: %d, the real answer is: %d" %(classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print ("the total error rate is: %f" %(errorCount/float(numTestVecs)))
    print (errorCount)

def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    ffMiles = float(input("frequent flier miles earned per year?"))
    percentTats = float(input("percentage of time spent playing video games?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print ("You will probably like this person: ", resultList[ classifierResult - 1])




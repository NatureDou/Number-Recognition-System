from numpy import *
import operator
from os import listdir
import matplotlib.pyplot as plt
import numpy as np
import random


#构建训练集数据向量，及对应分类标签向量
def trainingDataSet():
    hwLabels = []
    trainingFile = listdir('trainSet')           #获取目录内容
    length = len(trainingFile)
    trainingMat = zeros((length,1024))                         
    for i in range(length):
        fileName = trainingFile[i]
        hwLabels.append(getClassnum(fileName))
        trainingMat[i,:] = img2vector('trainSet/%s' % fileName)
    return hwLabels,trainingMat


#从训练集测试集文件中解析数字
def getClassnum(fileName): 
    fileStr = fileName.split('.')[0]  
    classNumStr = int(fileStr.split('_')[0]) 
    return classNumStr


#文本向量化 32x32 -> 1x1024
def img2vector(filename):
    returnVect = []
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect.append(int(lineStr[j]))
    return returnVect


#KNN算法实现
def classify(inputPoint,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]     #已知分类的数据集（训练集）的行数

    #计算距离
    diffMat = tile(inputPoint,(dataSetSize,1))-dataSet  
    sqDiffMat = diffMat ** 2                    
    sqDistances = sqDiffMat.sum(axis=1)         
    distances = sqDistances ** 0.5

    
    sortedDistIndicies = distances.argsort()    #进行升序排序

    #选择距离最小的k个点
    classCount = {}
    for i in range(k):
        voteIlabel = labels[ sortedDistIndicies[i] ]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1

    #获取距离最小点序列中占比最多的那个类并返回
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]


if __name__ == "__main__":        #主函数
    hwLabels,trainingMat = trainingDataSet()    #构建训练集
    testFile = listdir('testSet')            #获取测试集
    errorCount = 0.0                            #错误数计数
    testLen = len(testFile)                 #测试集总样本数
    
    x = np.array([])    #散点图横坐标，
    y = np.array([])    #散点图纵坐标

    print("正在运行中，请耐心等待")
       
    for i in range(testLen):
        fileNameStr = testFile[i]
        classNumStr = getClassnum(fileNameStr)
        vectorUnderTest = img2vector('testSet/%s' % fileNameStr)#调用knn算法进行测试
        classifierResult = classify(vectorUnderTest, trainingMat, hwLabels, 3)
        x = np.append(x,classifierResult+random.uniform(-0.1,0.1))   #增加随机数使得数量对比更加明显
        y = np.append(y,classNumStr+random.uniform(-0.1,0.1))
        if (classifierResult != classNumStr): errorCount += 1.0    #不匹配则增加错误计数值
        
    print ("\n测试总样本数为: %d" % testLen)               #输出测试总样本数
    print ("测试错误样本数为: %d" % errorCount)           #输出测试错误样本数
    print ("测试错误率为: %f" % (errorCount/float(testLen)))  #输出错误率

    plt.subplot(1,1,1)           #画出散点图
    plt.scatter(x,y,c = 'r')
    plt.show()

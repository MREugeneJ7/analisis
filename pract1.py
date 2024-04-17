from datetime import timedelta
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from matplotlib.colors import ListedColormap

def noFieldEmpty(array):
    for field in array:
        if field == '':
            return False
    return True

def dropUnnecessaryElements(dataframe): #TODO: Add chisquare for categorical variables? Will do if time allows
    dataframe = dataframe.drop("Loan_ID", axis=1)
    corrDataframe = dataframe.drop(["Gender", "Education", "PropertyArea"], axis=1)
    correlationMatrix = corrDataframe.corr(method='pearson', min_periods=1, numeric_only=False)
    correlationMatrix = pd.DataFrame(np.tril(correlationMatrix, k=-1))
    positiveCorrelatedVariables = correlationMatrix[correlationMatrix.gt(0.5)]
    positiveCorrelatedVariables = positiveCorrelatedVariables[correlationMatrix.ne(1)].any()
    for idx, a in enumerate(positiveCorrelatedVariables):
        if(a == True):
            dataframe = dataframe.drop(corrDataframe.columns[idx], axis=1)
    negativeCorrelatedVariables = correlationMatrix[correlationMatrix.lt(-0.5)]
    negativeCorrelatedVariables = negativeCorrelatedVariables[correlationMatrix.ne(1)].any()
    for idx, a in enumerate(negativeCorrelatedVariables):
        if(a == True):
            dataframe = dataframe.drop(corrDataframe.columns[idx], axis=1)
    return dataframe

def convertTypes(matrix):
    for vector in matrix:
        vector[headers.index('Married')] = vector[headers.index('Married')] != 'No'
        vector[headers.index('Dependents')] = int(vector[headers.index('Dependents')].replace("+", ""))
        vector[headers.index('SelfEmployed')] = vector[headers.index('SelfEmployed')] != 'No'
        vector[headers.index('ApplicantIncome')] = int(vector[headers.index('ApplicantIncome')])
        vector[headers.index('CoapplicantIncome')] = int(vector[headers.index('CoapplicantIncome')].replace(".", ""))
        vector[headers.index('LoanAmount')] = int(vector[headers.index('LoanAmount')])
        vector[headers.index('LoanAmountTerm')] = timedelta( days = int(vector[headers.index('LoanAmountTerm')]))
        vector[headers.index('LoanStatus')] = vector[headers.index('LoanStatus')] != 'N'
    print(matrix)
    return matrix

def deleteOutliers(dataFrame : pd.DataFrame):
    min = 0.01
    max = 0.99
    dependentOutlierLimits = dataFrame['Dependents'].quantile([min, max])
    dataFrame = dataFrame[dataFrame['Dependents'] >= dependentOutlierLimits[min]]
    dataFrame = dataFrame[dataFrame['Dependents'] <= dependentOutlierLimits[max]]
    applicantIncomeOutlierLimits = dataFrame['ApplicantIncome'].quantile([min, max])
    dataFrame = dataFrame[dataFrame['ApplicantIncome'] >= applicantIncomeOutlierLimits[min]]
    dataFrame = dataFrame[dataFrame['ApplicantIncome'] <= applicantIncomeOutlierLimits[max]]
    coapplicantIncomeOutlierLimits = dataFrame['CoapplicantIncome'].quantile([min, max])
    dataFrame = dataFrame[dataFrame['CoapplicantIncome'] >= coapplicantIncomeOutlierLimits[min]]
    dataFrame = dataFrame[dataFrame['CoapplicantIncome'] <= coapplicantIncomeOutlierLimits[max]]
    loanAmountOutlierLimits = dataFrame['LoanAmount'].quantile([min, max])
    dataFrame = dataFrame[dataFrame['LoanAmount'] >= loanAmountOutlierLimits[min]]
    dataFrame = dataFrame[dataFrame['LoanAmount'] <= loanAmountOutlierLimits[max]]
    loanAmountTermOutlierLimits = dataFrame['LoanAmountTerm'].quantile([min, max])
    dataFrame = dataFrame[dataFrame['LoanAmountTerm'] >= loanAmountTermOutlierLimits[min]]
    dataFrame = dataFrame[dataFrame['LoanAmountTerm'] <= loanAmountTermOutlierLimits[max]]
    return dataFrame

def overSampleForBalance(dataFrame):
    y = dataFrame["LoanStatus"]
    ratio, trueCount, falseCount = 0,0,0
    for count in y.value_counts():
        if(trueCount == 0):
            trueCount = count
        else:
            falseCount = count
    if(trueCount < falseCount):
        ratio = falseCount/trueCount
        repeatedDf = pd.DataFrame(np.repeat(dataFrame[dataFrame["LoanStatus"] == True].values, ratio - 1 , axis=0), columns = dataFrame.columns)
        sampleDf = dataFrame[dataFrame["LoanStatus"] == True].sample(frac = ratio - int(ratio))
        dataFrame = pd.concat([dataFrame, repeatedDf, sampleDf])
    else:
        ratio = trueCount/falseCount
        repeatedDf = pd.DataFrame(np.repeat(dataFrame[dataFrame["LoanStatus"] == False].values, ratio - 1 , axis=0), columns = dataFrame.columns)
        sampleDf = dataFrame[dataFrame["LoanStatus"] == False].sample(frac = ratio - int(ratio))
        dataFrame = pd.concat([dataFrame, repeatedDf, sampleDf])

    y = dataFrame["LoanStatus"]
    ratio, trueCount, falseCount = 0,0,0
    for count in y.value_counts():
        if(trueCount == 0):
            trueCount = count
        else:
            falseCount = count
    print(trueCount)
    print(falseCount)
    return dataFrame

data = open('./homeLoanAproval.csv', 'r')
headers = data.readline().replace("\n", "").split(",")
print(headers)
body = []
for line in data:
    body.append(line.replace("\n", "").split(","))

body = list(filter(noFieldEmpty, body))

body = convertTypes(body)

body = np.array(body)
bodyDf = pd.DataFrame(body, columns = headers)


bodyDf = deleteOutliers(bodyDf)


bodyDf = dropUnnecessaryElements(bodyDf)

trainingDf = bodyDf.sample(frac=0.7)
evalDf = pd.concat([bodyDf,trainingDf]).drop_duplicates(keep=False)

trainingDf = overSampleForBalance(trainingDf)

#Prepare the dataframe to fuck it up

le = LabelEncoder()
le.fit(pd.concat([trainingDf["Gender"], trainingDf["Education"], trainingDf["PropertyArea"]]))
trainingDf["Gender"] = le.transform(trainingDf["Gender"])
trainingDf["Education"] = le.transform(trainingDf["Education"])
trainingDf["PropertyArea"] = le.transform(trainingDf["PropertyArea"])
trainingDf["LoanAmountTerm"] = trainingDf["LoanAmountTerm"].map( lambda x : x.total_seconds())

evalDf["Gender"] = le.transform(evalDf["Gender"])
evalDf["Education"] = le.transform(evalDf["Education"])
evalDf["PropertyArea"] = le.transform(evalDf["PropertyArea"])
evalDf["LoanAmountTerm"] = evalDf["LoanAmountTerm"].map( lambda x : x.total_seconds())


#KNN

n = KNeighborsClassifier(3)
n.fit(trainingDf.drop("LoanStatus", axis=1), trainingDf["LoanStatus"].map(lambda x : str(x)))

score = n.score(evalDf.drop("LoanStatus", axis=1), evalDf["LoanStatus"].map(lambda x : str(x)))

print(score)

#Arbol

clf = tree.DecisionTreeClassifier(max_depth=5)
clf = clf.fit(trainingDf.drop("LoanStatus", axis=1), trainingDf["LoanStatus"].map(lambda x : str(x)))
treeScore = clf.score(evalDf.drop("LoanStatus", axis=1), evalDf["LoanStatus"].map(lambda x : str(x)))

print(treeScore)

tree.plot_tree(clf)

plt.show()

#Naive BAyes
gnb = GaussianNB()
gnb = gnb.fit(trainingDf.drop("LoanStatus", axis=1), trainingDf["LoanStatus"].map(lambda x : str(x)))
naiveScore = gnb.score(evalDf.drop("LoanStatus", axis=1), evalDf["LoanStatus"].map(lambda x : str(x)))

print(naiveScore)

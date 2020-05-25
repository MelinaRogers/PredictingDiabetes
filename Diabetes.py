import pandas as pd #Reads data from CSV and manipulates it
import numpy as np #convert out data and make it suitable for classification model
import seaborn as sns #Used for visualizations
import matplotlib.pyplot as plt #Used for visualizations
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

warnings.filterwarnings('ignore')

diabetesDF = pd.read_csv('diabetes.csv') #Read the dataset into pandas
#print(diabetesDF.head()) Show first five records from the dataset

'''Finding correlation of every pair of features (and the outcome
variable) and visualize the correlations using a heatmap'''

corr = diabetesDF.corr()
sns.heatmap(corr, xticklabels=corr.columns,yticklabels=corr.columns)

'''Dataset Preparation '''

dfTrain = diabetesDF[:650]
dfTest = diabetesDF[650:750]
dfCheck = diabetesDF[750:]

'''Separate the label and features (for both training and test
datasets) Also converting them into NumPy arrays so that the ML
algorithm can process them'''

trainLabel = np.asarray(dfTrain['Outcome'])
trainData = np.asarray(dfTrain.drop('Outcome',1)) #drops specified labels from rows or columns
testLabel = np.asarray(dfTest['Outcome'])
testData = np.asarray(dfTest.drop('Outcome', 1))

train_test = dfTrain.drop('Outcome',1)


'''Normalize the inputs'''

means = np.mean(trainData, axis = 0)
stds = np.std(trainData, axis = 0)

trainData = (trainData - means) / stds
testData = (testData - means) / stds


'''Training classification model. This is logistic regression since
its already available in sklearn'''

diabetesCheck = LogisticRegression() #Creates an instance
diabetesCheck.fit(trainData, trainLabel) #fit function trains the model

'''Use test data to find out the accuracy of the model'''

accuracy = diabetesCheck.score(testData,testLabel)
#Finds the difference between the fitted Y -values and what your model found ^^^
print("accuracy = ", accuracy * 100, "%")

'''Interpreting the ML Model'''

coeff = list(diabetesCheck.coef_[0])
labels = list(train_test.columns)
features = pd.DataFrame()
features['Features'] = labels
features['importance'] = coeff
features.sort_values(by=['importance'], ascending=True, inplace=True)
features['positive'] = features['importance'] > 0
features.set_index('Features', inplace=True)
features.importance.plot(kind='barh', figsize=(11, 6),color = features.positive.map({True: 'blue', False: 'red'}))
plt.xlabel('Importance')

'''Saving the Model'''

joblib.dump([diabetesCheck, means, stds], 'diabeteseModel.pkl')

diabetesLoadedModel, means, stds = joblib.load('diabeteseModel.pkl')
accuracyModel = diabetesLoadedModel.score(testData, testLabel)


'''Making Predictions'''
#print(dfCheck.head()) #dfCheck holds the unused data

#Now we can use the first record to make the prediction
sampleData = dfCheck[:1]

#Prepare sample
sampleDataFeatures = np.asarray(sampleData.drop('Outcome', 1))
sampleDataFeatures = (sampleDataFeatures - means) / stds

#Make prediction
predictionProbability = diabetesLoadedModel.predict_proba(sampleDataFeatures)
prediction = diabetesLoadedModel.predict(sampleDataFeatures)
#print('Probability: ', predictionProbability) #first element is probability of 0 and second is probability of being 1
#print('prediction: ', prediction)



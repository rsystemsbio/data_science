# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 13:45:15 2020

@author: Rachel Calder

Data Preperation:
The following data frame (the Adult dataset from the UCI machine learning databse 
provided by Ronny Kohavi and Barry Becker and can be found at
https://archive.ics.uci.edu/ml/datasets/Adult)

has 15 attributes with numerical and categorical data. Missing data and outlier 
Numeric 'hours-per-week' column was normalized using the zscaling method and binned 
using equal width  binning with 5 bins (determined after histogram visualization. 
Categorical variable 'education-num' was decoded using the numerical 'education' 
column values. Within this 9th and 10th and 11th and 12th 'education-num' values 
were consolidated to be consistent with previous grade groupings. The 'education-num' 
variable was then one-hot encoded. Missing values were found in categorial variables 
'native-country', 'workclass', and 'occupation' and were imputed. 'native-country' 
was plotted to ensure no missing variables persisted. Outliers were found in and 
removed from 'age', 'fnlwgt', 'capital gain', and 'capital loss' and replaced with
attribute mean.

Feature description:
    Feature       data type       distribution         additional comments
    age           float64           Normal             outliers removed
workclass          object           -                  missing values imputed
fnlwgt            float64           Normal             outliers removed
education          object           -                  used for binning
marital-status     object           -                  -
occupation         object           -                  missing values imputed
relationship       object           -                  -
race               object           -                  -
sex                object           -                  -
capital-gain      float64           Positive Skew      outliers removed
capital-loss      float64           Positive Skew      outliers removed
hours-per-week      int64           Normal             missing values imputed
native-country     object           -                  missing values imputed
income             object           -                  -
Bachelors           int64           Bimodal            from One-hot encoded education-num
HS-grad             int64           Bimodal            from One-hot encoded education-num
11th-12th           int64           Bimodal            from One-hot encoded education-num
Masters             int64           Bimodal            from One-hot encoded education-num
9th-10th            int64           Bimodal            from One-hot encoded education-num
Some-college        int64           Bimodal            from One-hot encoded education-num
Assoc-acdm          int64           Bimodal            from One-hot encoded education-num
Assoc-voc           int64           Bimodal            from One-hot encoded education-num
7th-8th             int64           Bimodal            from One-hot encoded education-num
Doctorate           int64           Bimodal            from One-hot encoded education-num
Prof-school         int64           Bimodal            from One-hot encoded education-num
5th-6th             int64           Bimodal            from One-hot encoded education-num
1st-4th             int64           Bimodal            from One-hot encoded education-num
Preschool           int64           Bimodal            from One-hot encoded education-num
Cluster Label       int64           Bimodal            Kmeans cluster of age and one-hot
                                                          encoded education-num
Machine Learning Analysis and Results:
A Logistic Regression classifier (LR) and a Support Vector Machine classifier (SVM) 
were used to predict if a person made more than $50K a year or not. In comparing 
precision, accuracy, AUC score, and F1 score, the classifiers perform similarly.
While LR is more percise, SVM has a higher AUC score, and F1 score. The classifiers
have the same accuracy. The results suggest potentially leaning more toward using
the SVM classifier, even though LR is more percise for this target, the F1 score,
which precision is used to calculate, is higher for the SVM classifier.

Measure           LR       SVM 
AUC Score(W/ ROC) 0.58     0.59    
Precision         0.8      0.7   
Accuracy          0.79     0.79
F1 Score          0.24     0.27

"""
#Import necessary packages (#1)
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.svm import SVC


url="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
pd.options.mode.chained_assignment = None
#Read in Data from the online archive (#2)
ad = pd.read_csv(url,error_bad_lines=False)
#assign column names
ad.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
               'marital-status', 'occupation', 'relationship', 'race',
                'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 
                'native-country','income']


"""Data Cleaning Before Machine Learning"""

"""Normalizing 'hours-per-week"""
#Numpy normalization (#3)
x = ad.iloc[:, 12].values
minmaxscaled =(x - min(x))/(max(x) - min(x))
zscaled = (x - np.mean(x))/np.std(x)
print ("\nScaled variable x using numpy calculations\n")
print(np.hstack(
        (np.reshape(x,(32560,1)),
         np.reshape(minmaxscaled,(32560,1)),
         np.reshape(zscaled, (32560,1)))
          ))

#checking appropriate # of bins
sns.distplot(x)

#Binning of hours per week (#4)
NumberOfBins = 5
BinWidth = (max(x) - min(x))/NumberOfBins
MinBin1 = float('-inf')
MaxBin1 = min(x) + 1 * BinWidth
MaxBin2 = min(x) + 2 * BinWidth
MaxBin3 = min(x) + 3 * BinWidth
MaxBin4 = min(x) + 4 * BinWidth
MaxBin5 = float('inf')


print("Bin 1 ends at",BinWidth)
print("Bin 2 ends at",MaxBin2)
print("Bin 3 ends at",MaxBin3)
print("Bin 4 ends at",MaxBin4)
print("Bin 5 ends at",MaxBin5)

xBinnedEqW = np.empty(len(x), object) # np.full(len(x), "    ")

# The conditions at the boundaries should consider the difference 

xBinnedEqW[(x > MinBin1) & (x <= MaxBin1)] = "Very Low"
xBinnedEqW[(x > MaxBin1) & (x <= MaxBin2)] = "Low"
xBinnedEqW[(x > MaxBin2) & (x <= MaxBin3)] = "Med"
xBinnedEqW[(x > MaxBin3) & (x <= MaxBin4)] = "High"
xBinnedEqW[(x > MaxBin4) & (x <= MaxBin5)] = "Very High"
print(" x binned into 5 equal-width bins:", xBinnedEqW)


"""Decoding, Consolidation, and One-hot encoding  of education-num"""
#Decoding education-num and consolidation (#5, #7)
ad.loc[ad.loc[:, "education-num"] == 13, "education-num"] = "Bachelors"
ad.loc[ad.loc[:, "education-num"] == 9, "education-num"] = "HS-grad"
ad.loc[ad.loc[:, "education-num"] == 7, "education-num"] = "11th-12th"
ad.loc[ad.loc[:, "education-num"] == 14, "education-num"] = "Masters"
ad.loc[ad.loc[:, "education-num"] == 5, "education-num"] = "9th-10th"
ad.loc[ad.loc[:, "education-num"] == 10, "education-num"] = "Some-college"
ad.loc[ad.loc[:, "education-num"] == 12, "education-num"] = "Assoc-acdm"
ad.loc[ad.loc[:, "education-num"] == 11, "education-num"] = "Assoc-voc"
ad.loc[ad.loc[:, "education-num"] == 4, "education-num"] = "7th-8th"
ad.loc[ad.loc[:, "education-num"] == 16, "education-num"] = "Doctorate"
ad.loc[ad.loc[:, "education-num"] == 15, "education-num"] = "Prof-school"
ad.loc[ad.loc[:, "education-num"] == 3, "education-num"] = "5th-6th"
ad.loc[ad.loc[:, "education-num"] == 6, "education-num"] = "9th-10th"
ad.loc[ad.loc[:, "education-num"] == 2, "education-num"] = "1st-4th"
ad.loc[ad.loc[:, "education-num"] == 1, "education-num"] = "Preschool"
ad.loc[ad.loc[:, "education-num"] == 8, "education-num"] = "11th-12th"

#Get the counts for each value
ad.loc[:,"education-num"].value_counts()

# Create 14 new columns, one for each state in "education-num" #4
ad.loc[:, "Bachelors"] = (ad.loc[:, "education-num"] == "Bachelors").astype(int)
ad.loc[:, "HS-grad"] = (ad.loc[:, "education-num"] == "HS-grad").astype(int)
ad.loc[:, "11th-12th"] = (ad.loc[:, "education-num"] == "11th-12th").astype(int)
ad.loc[:, "Masters"] = (ad.loc[:, "education-num"] == "Masters").astype(int)
ad.loc[:, "9th-10th"] = (ad.loc[:, "education-num"] == "9th-10th").astype(int)
ad.loc[:, "Some-college"] = (ad.loc[:, "education-num"] == "Some-college").astype(int)
ad.loc[:, "Assoc-acdm"] = (ad.loc[:, "education-num"] == "Assoc-acdm").astype(int)
ad.loc[:, "Assoc-voc"] = (ad.loc[:, "education-num"] == "Assoc-voc").astype(int)
ad.loc[:, "7th-8th"] = (ad.loc[:, "education-num"] == "7th-8th").astype(int)
ad.loc[:, "Doctorate"] = (ad.loc[:, "education-num"] == "Doctorate").astype(int)
ad.loc[:, "Prof-school"] = (ad.loc[:, "education-num"] == "Prof-school").astype(int)
ad.loc[:, "5th-6th"] = (ad.loc[:, "education-num"] == "5th-6th").astype(int)
ad.loc[:, "1st-4th"] = (ad.loc[:, "education-num"] == "1st-4th").astype(int)
ad.loc[:, "Preschool"] = (ad.loc[:, "education-num"] == "Preschool").astype(int)
##############

# Remove obsolete column #5
ad = ad.drop("education-num", axis=1)
# Check new dataset
ad.head()
ad.dtypes

"""Impute missing values for categorical columns"""
# Check for missing values
ad.loc[:,"native-country"].value_counts()

# Create a boolean for the missing value
MissingValues = ad.loc[:, "native-country"] == " ?"

# Impute missing value in native-country
ad.loc[ad.loc[:, "native-country"] == " ?", "native-country"] = "United-States"


#repeat for other categorial columns
ad.loc[:,"workclass"].value_counts()
MissingValues = ad.loc[:, "workclass"] == " ?"
ad.loc[ad.loc[:, "workclass"] == " ?", "workclass"] = "Private"

ad.loc[:,"occupation"].value_counts()
MissingValues = ad.loc[:, "occupation"] == " ?"
ad.loc[ad.loc[:, "occupation"] == " ?", "occupation"] = "Prof-specialty"

"""Outliers removed for numerical columns"""
numer_cols = ["fnlwgt", "age", "capital-gain", "capital-loss"]
for item in numer_cols:
#Establishes the low and high outlier limits as two standard deviations
    LimitHi = np.mean(ad.loc[:,item]) + 2*np.std(ad.loc[:,item])
    LimitLo = np.mean(ad.loc[:,item]) - 2*np.std(ad.loc[:,item])
        #Establishes that values within two standard deviations are good
    FlagGood = (ad.loc[:,item] >= LimitLo) & (ad.loc[:,item] <= LimitHi)
    FlagBad = ~FlagGood
    ad.loc[:,item][FlagBad] = np.mean(ad.loc[:,item])


""" Machine Learning for age and one-hot encoded education-num"""
# Create Points to cluster
Points = pd.DataFrame()
Points.loc[:,0] = ad.loc[:,"age"]
Points.loc[:,1] = ad.loc[:, "Bachelors"] 
Points.loc[:,2] = ad.loc[:, "11th-12th"]
Points.loc[:,3] = ad.loc[:, "Masters"]
Points.loc[:,4] = ad.loc[:, "9th-10th"]
Points.loc[:,5] = ad.loc[:, "Some-college"]
Points.loc[:,6] = ad.loc[:, "Assoc-acdm"]
Points.loc[:,7] = ad.loc[:, "Assoc-voc"]
Points.loc[:,8] = ad.loc[:, "7th-8th"]
Points.loc[:,9] = ad.loc[:, "Doctorate"]
Points.loc[:,10] = ad.loc[:, "Prof-school"]
Points.loc[:,11] = ad.loc[:, "5th-6th"]
Points.loc[:,12] = ad.loc[:, "1st-4th"]
Points.loc[:,13] = ad.loc[:, "Preschool"]

# Create initial cluster centroids
ClusterCentroidGuesses = pd.DataFrame()
ClusterCentroidGuesses.loc[:,0] = [40, 50]
for i in range(1,14):
    ClusterCentroidGuesses.loc[:,i] = [1, 1]


#One-hot encode remaining columns
#Create dictionary
ad = ad.drop('education',1)
columnNames = ["workclass", "marital-status", "occupation", 
                   "relationship", "race", "sex", "native-country", "income"]
#Column: Values

def Plot2DKMeans(Points, Labels, ClusterCentroids, Title):
    for LabelNumber in range(max(Labels)+1):
        LabelFlag = Labels == LabelNumber
        color =  ['c', 'm', 'y', 'b', 'g', 'r', 'c', 'm', 'y',
                  'b', 'g', 'r', 'c', 'm', 'y'][LabelNumber]
        marker = ['s', 'o', 'v', '^', '<', '>', '8', 'p', '*', 
                  'h', 'H', 'D', 'd', 'P', 'X'][LabelNumber]
        plt.scatter(Points.loc[LabelFlag,0], Points.loc[LabelFlag,1],
                    s= 100, c=color, edgecolors="black", alpha=0.3, marker=marker)
        plt.scatter(ClusterCentroids.loc[LabelNumber,0],
                    ClusterCentroids.loc[LabelNumber,1], 
                    s=200, c="black", marker=marker)
    plt.title(Title)
    plt.show()

def KMeansNorm(Points, ClusterCentroidGuesses, input_norms):
    PointsNorm = Points.copy()
    ClusterCentroids = ClusterCentroidGuesses.copy()
    for i in range(len(input_norms)):
        means = []
        stds = []
        cmeans = []
        cstds = []
        if input_norms[i]:
            # Determine mean of 1st dimension
            val = PointsNorm.iloc[:,i]
            mean=np.mean(val)
            means.append(mean)
            # Determine standard deviation of 1st dimension
            std = np.std(val)
            stds.append(std)
            # Normalize 1st dimension of Points
            val = ((val - mean)/std) 
            # Normalize 1st dimension of ClusterCentroids
            Cmean = np.mean(ClusterCentroids[i])
            cmeans.append(Cmean)
            Cstd = np.std(ClusterCentroids[i])
            cstds.append(Cstd)
            ClusterCentroids[i] = ((ClusterCentroids[i] - Cmean)/Cstd)
            
        # Do actual clustering
        kmeans = KMeans(n_clusters=2, init=ClusterCentroidGuesses, n_init=1).fit(PointsNorm)
        Labels = kmeans.labels_
        ClusterCentroids = pd.DataFrame(kmeans.cluster_centers_)
        if input_norms[i]:
            # Denormalize dimension
            PointsNorm[i] = PointsNorm[i]*stds[i]+means[i]
            ClusterCentroids[i] = ClusterCentroids[i]*cstds[i]+cmeans[i]
 
    return Labels, ClusterCentroids

# Compare distributions of the two dimensions
plt.rcParams["figure.figsize"] = [6.0, 4.0] # Standard
plt.hist(Points.loc[:,0], bins = 20, color=[0, 0, 1, 0.5])
plt.hist(Points.loc[:,1], bins = 20, color=[1, 1, 0, 0.5])
plt.hist(Points.loc[:,2], bins = 20, color=[1, 1, 0, 0])
plt.hist(Points.loc[:,3], bins = 20, color=[1, 0.5, 0, 0.5])
plt.hist(Points.loc[:,4], bins = 20, color=[1, 0.5, 0, 0])
plt.hist(Points.loc[:,5], bins = 20, color=[1, 0.5, 0, 0.5])
plt.hist(Points.loc[:,6], bins = 20, color=[1, 0, 0, 0.5])
plt.hist(Points.loc[:,7], bins = 20, color=[1, 1, 1, 0.5])
plt.hist(Points.loc[:,8], bins = 20, color=[1, 1, 0.2, 0.5])
plt.hist(Points.loc[:,9], bins = 20, color=[1, 1, 0, 0.5])
plt.hist(Points.loc[:,10],bins = 20, color=[0.2, 1, 0, 0.5])
plt.hist(Points.loc[:,11], bins = 20, color=[1, 1, 0, 0.2])
plt.hist(Points.loc[:,12], bins = 20, color=[1, 0.2, 0, 0.5])
plt.hist(Points.loc[:,13], bins = 20, color=[1, 1, 0.2, 0.5])
plt.title("Compare Distributions")
plt.show()

# Change the plot dimensions
plt.rcParams["figure.figsize"] = [8, 8] # Square

# Cluster with both dimensions normalized 
NormD1=True
NormD2=True
NormD3=True
NormD4=True
NormD5=True
NormD6=True
NormD7=True
NormD8=True
NormD9=True
NormD10=True
NormD11=True
NormD12=True
NormD13=True
NormD14=True

input_norms = [NormD1, NormD2, NormD3, NormD4, NormD5, NormD6, NormD7,NormD8, NormD9, NormD10, NormD11, NormD12,NormD13, NormD14]
Labels, ClusterCentroids = KMeansNorm(Points, ClusterCentroidGuesses, input_norms)
Title = 'Normalized, Clustered '
Plot2DKMeans(Points, Labels, ClusterCentroids, Title)

"""
The One hot encoded education-num columns and age attributes cluster into two
clusters. The cluster is indicated in the Cluster Label column as a zero or a 1. 
"""
#add labels to dataframe
ad['Cluster Label'] = Labels.tolist()


def OneHotEncoing(categories:list, currentColumn:str):
    """
    This will create One-Hot encoding for a categorical list
    """
    
    for item in categories:
        print("Current Value: ", item)
        #Create column and add values
        ad.loc[:, item] = (ad.loc[:, currentColumn] == item).astype(int)
        
    return None


def start():
    """
    Iterate through data set and add One-Hot columns
    """
    for x in range(len(columnNames)): 
        #Get current column
        currentColumn = columnNames[x]
        print("Column name: ", currentColumn)
        
        #get unique values
        uniqueValues = list(ad[columnNames[x]].unique())
        OneHotEncoing(uniqueValues, currentColumn)
    
    return None

#Function call
start()
#Dropping obsolete columna
ad = ad.drop("workclass",1)
ad = ad.drop('marital-status',1)
ad = ad.drop('occupation',1)
ad = ad.drop('relationship',1)
ad = ad.drop('race',1)
ad = ad.drop('sex',1)
ad = ad.drop('native-country',1)
ad = ad.drop('income',1)

#making dataframe to use for dataframe labeling
class_free = ad.drop(' >50K',1)
"""
Question: Does this person make more than $50K annually?
Expert label column: >50K
"""
#Assigning data and target labels
columnLen = len(ad.columns)
data = ad.iloc[: , :columnLen-1]
target = ad.iloc[: , columnLen-1]
data = data.values
target = target.values

#using appropriate sklearn function to split data into train and test sets
X, XX, Y, YY = train_test_split(data, target, train_size = .9)

#X = data test, XX = data train, Y = class data, YY = class train


""" Logistic Regression Classifier """


# Logistic regression classifier
print ('\n\n\nLogistic regression classifier\n')
C_parameter = 50. / len(X) # parameter for regularization of the model
class_parameter = 'ovr' # parameter for dealing with multiple classes
penalty_parameter = 'l1' # parameter for the optimizer (solver) in the function
solver_parameter = 'saga' # optimization system used
tolerance_parameter = 0.1 # termination parameter
#####################

#Training the Model
clf = LogisticRegression(C=C_parameter, multi_class=class_parameter, penalty=penalty_parameter, solver=solver_parameter, tol=tolerance_parameter)
clf.fit(X, Y) 
print ('coefficients:')
print (clf.coef_) # each row of this matrix corresponds to each one of the classes of the dataset
print ('intercept:')
print (clf.intercept_) # each element of this vector corresponds to each one of the classes of the dataset

# Apply the Model
print ('predictions for test set:')
print (clf.predict(XX))
probs = clf.predict(XX)
print ('actual class values:')
print (YY)
#####################

#Turning array values into into a concatonated dataframe
X_df = pd.DataFrame(X)
X_df.columns = class_free.columns
YY_df = pd.DataFrame(YY)
YY_df.columns = ["actual_outcomes"]
probs_df = pd.DataFrame(probs)
probs_df.columns = ["probabilities"]
log_results = pd.DataFrame()
log_results = pd.concat([X_df, YY_df, probs_df], axis=1)

#wiriting results to a csv file
print ('writing results to csv file...')
log_results.to_csv('log_results_rc.csv')

#Accuracy rate
#Determining which guesses are correct by comparing values in a boolean
compare = YY == probs
#Since TRUE has a value of 1, summing our boolean over the total number of 
#predictions gives the accuracy
print ("Accuracy is" , sum(compare)/len(probs))

#defining variables from Logistic regression classifier for accuracy testing
T = YY
c = clf.predict_proba(XX)
y = c[:,1]
yr = np.round(y,0)

###################

# Confusion Matrix
CM = confusion_matrix(T, yr)
print ("\n\nConfusion matrix:\n", CM)
tn, fp, fn, tp = CM.ravel()
print ("\nTP, TN, FP, FN:", tp, ",", tn, ",", fp, ",", fn)
AR = accuracy_score(T, yr)
print ("\nAccuracy rate:", AR)
ER = 1.0 - AR
print ("\nError rate:", ER)
P = precision_score(T, yr)
print ("\nPrecision:", np.round(P, 2))
R = recall_score(T, yr)
print ("\nRecall:", np.round(R, 2))
F1 = f1_score(T, yr)
print ("\nF1 score:", np.round(F1, 2))
####################

# ROC analysis
LW = 1.5 # line width for plots
LL = "lower right" # legend location
LC = 'darkgreen' # Line Color

#probability threshold is created using the ROC curve
fpr, tpr, th = roc_curve(T, y) # False Positive Rate, True Posisive Rate, probability thresholds
#calculting AUC
AUC = auc(fpr, tpr)
print ("\nTP rates:", np.round(tpr, 2))
print ("\nFP rates:", np.round(fpr, 2))
print ("\nProbability thresholds:", np.round(th, 2))
#####################

#Presenting ROC Curve
plt.figure()
plt.title('Receiver Operating Characteristic curve Logistic Regression')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FALSE Positive Rate')
plt.ylabel('TRUE Positive Rate')
plt.plot(fpr, tpr, color=LC,lw=LW, label='ROC curve (area = %0.2f)' % AUC)
plt.plot([0, 1], [0, 1], color='navy', lw=LW, linestyle='--') # reference line for random classifier
plt.legend(loc=LL)
plt.show()
####################

#Presenting AUC Score
print ("\nAUC score (using auc function):", np.round(AUC, 2))
print ("\nAUC score (using roc_auc_score function):", np.round(roc_auc_score(T, y), 2), "\n")

######################
# Support vector machine classifier
t = 0.001 # tolerance parameter (default)
kp = 'rbf' # kernel parameter (default)
print ('\n\nSupport Vector Machine classifier\n')
clf = SVC(kernel=kp, tol=t, probability=True)
clf.fit(X, Y)
print ("predictions for test set:")
print (clf.predict(XX))
SVM_probs = clf.predict(XX)
print ('actual class values:')
print (YY)

#prepping dataframe for csv file
SVM_X_df = pd.DataFrame(X)
SVM_X_df.columns = class_free.columns
SVM_YY_df = pd.DataFrame(YY)
SVM_YY_df.columns = ["actual_outcomes"]
SVM_probs_df = pd.DataFrame(SVM_probs)
SVM_probs_df.columns = ["probabilities"]
SVM_log_results = pd.DataFrame()
SVM_log_results = pd.concat([SVM_X_df, SVM_YY_df, SVM_probs_df], axis=1)
#wiriting results to a csv file
print ('writing results to csv file...')
SVM_log_results.to_csv('SVM_results_rc.csv')

#Accuracy rate
#Determining which guesses are correct by comparing values in a boolean
compare = YY == SVM_probs
#Since TRUE has a value of 1, summing our boolean over the total number of 
#predictions gives the accuracy
print ("Accuracy is" , sum(compare)/len(probs))

#defining variables from SVM classifier for accuracy testing
T = YY
c = clf.predict_proba(XX)
y = c[:,1]
yr = np.round(y,0)

###################

# Confusion Matrix
CM = confusion_matrix(T, yr)
print ("\n\nConfusion matrix:\n", CM)
tn, fp, fn, tp = CM.ravel()
print ("\nTP, TN, FP, FN:", tp, ",", tn, ",", fp, ",", fn)
AR = accuracy_score(T, yr)
print ("\nAccuracy rate:", AR)
ER = 1.0 - AR
print ("\nError rate:", ER)
P = precision_score(T, yr)
print ("\nPrecision:", np.round(P, 2))
R = recall_score(T, yr)
print ("\nRecall:", np.round(R, 2))
F1 = f1_score(T, yr)
print ("\nF1 score:", np.round(F1, 2))
####################

# ROC analysis
LW = 1.5 # line width for plots
LL = "lower right" # legend location
LC = 'darkgreen' # Line Color

#probability threshold is created using the ROC curve
fpr, tpr, th = roc_curve(T, y) # False Positive Rate, True Posisive Rate, probability thresholds
#calculting AUC
AUC = auc(fpr, tpr)
print ("\nTP rates:", np.round(tpr, 2))
print ("\nFP rates:", np.round(fpr, 2))
print ("\nProbability thresholds:", np.round(th, 2))
#####################

#Presenting ROC Curve
plt.figure()
plt.title('Receiver Operating Characteristic curve for SVM')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FALSE Positive Rate')
plt.ylabel('TRUE Positive Rate')
plt.plot(fpr, tpr, color=LC,lw=LW, label='SVM ROC curve (area = %0.2f)' % AUC)
plt.plot([0, 1], [0, 1], color='navy', lw=LW, linestyle='--') # reference line for random classifier
plt.legend(loc=LL)
plt.show()
####################

#Presenting AUC Score
print ("\n SVM AUC score (using auc function):", np.round(AUC, 2))
print ("\n SVM AUC score (using roc_auc_score function):", np.round(roc_auc_score(T, y), 2), "\n")









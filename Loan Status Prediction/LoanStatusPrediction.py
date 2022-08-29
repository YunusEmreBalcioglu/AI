# libs
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# Data loading
loan_dataset=pd.read_csv("F:/ML projects/loandata.csv")


# Number of missing values
numberofmissing=loan_dataset.isnull().sum()

# Drop missing values
loan_dataset=loan_dataset.dropna()

# Label encoding
loan_dataset.replace({"Loan_Status":{"N":0,"Y":1}},inplace=True)

#Check data
print(loan_dataset["Dependents"].value_counts())
loan_dataset.replace({"Dependents":{"3+":4}},inplace=True)
print(loan_dataset["Dependents"].value_counts())

sns.countplot(x="Education",hue="Loan_Status",data=loan_dataset)
#plt.show()

sns.countplot(x="Married",hue="Loan_Status",data=loan_dataset)
#plt.show()

# Convert categorical columns to numerical values
loan_dataset.replace({"Married":{"No":0,"Yes":1},"Gender":{"Male":1,"Female":0},"Self_Employed":{"No":0,"Yes":1},
                      "Property_Area":{"Rural":0,"Urban":1,"Semiurban":2},"Education":{"Graduate":1,"Not Graduate":0}}
                     ,inplace=True)
#Separating data X and labe Y
X=loan_dataset.drop(columns=["Loan_ID","Loan_Status"],axis=1)
Y=loan_dataset["Loan_Status"]

# Split train and test data

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=2)

print(X.shape,X_test.shape,X_train.shape)

# Train Model with SVM
classifier=svm.SVC(kernel="linear")
classifier.fit(X_train,Y_train)

# Model Evalution
Y_train_prime=classifier.predict(X_train)
training_data_accuracy=accuracy_score(Y_train_prime,Y_train)
print(training_data_accuracy)

Y_test_prime=classifier.predict(X_test)
test_data_accuracy=accuracy_score(Y_test_prime,Y_test)
print(test_data_accuracy)

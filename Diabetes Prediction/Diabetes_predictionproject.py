import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# loading data set to pandas frame
diabetes_data=pd.read_csv("E:/ML projects/diabetes.csv")
print(diabetes_data.groupby("Outcome").mean())

X=diabetes_data.drop(columns="Outcome", axis=1)
Y=diabetes_data["Outcome"]
X_standardized=StandardScaler().fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X_standardized,Y, test_size=0.1, stratify=Y, random_state=2)

classifier=svm.SVC(kernel="linear")
classifier.fit(X_train, Y_train)

train_prediction=classifier.predict(X_train)
trainnig_accuracy=accuracy_score(train_prediction,Y_train)

print("Trainnig Accuracy: ", trainnig_accuracy)

test_prediction=classifier.predict(X_test)
test_accuracy=accuracy_score(test_prediction,Y_test)

print("Test Accuracy: ", test_accuracy)

input=np.asarray((0,137,40,35,168,43.1,2.288,33))
input_reshaped=input.reshape(1,-1)

print(input_reshaped)

prediction=classifier.predict(input_reshaped)
print(prediction)
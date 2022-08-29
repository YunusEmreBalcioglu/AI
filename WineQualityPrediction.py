# Libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Data loading
winequality_dataset=pd.read_csv("F:/ML projects/winequality.csv")

print(winequality_dataset.shape)
print(winequality_dataset.head())
print(winequality_dataset.value_counts("quality"))
print(winequality_dataset.isnull().sum())
print(winequality_dataset.describe())

# data visualization
plot= plt.figure(figsize=(5,5))
sns.catplot(x="quality",data=winequality_dataset,kind="count")
#plt.show()

sns.barplot(x="quality",y="volatile acidity",data=winequality_dataset)
#plt.show()

sns.barplot(x="quality",y="citric acid",data=winequality_dataset)
#plt.show()

correlation=winequality_dataset.corr()
plot= plt.figure(figsize=(10,10))
sns.heatmap(correlation,cbar=True,square=True,fmt=".1f",annot=True, annot_kws={"size":8},cmap="Blues")
#plt.show()

# Data spliting
X=winequality_dataset.drop("quality",axis=1)
Y=winequality_dataset["quality"].apply(lambda y_value:1 if y_value>=7 else 0)
print(Y.value_counts())

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)
print(Y.shape,Y_test.shape,Y_train.shape)

#Model training

model = RandomForestClassifier()
model.fit(X_train,Y_train)

#Model Evaluations

Y_train_prime=model.predict(X_train)
trainnin_accuracy=accuracy_score(Y_train,Y_train_prime)
print(trainnin_accuracy)

Y_test_prime=model.predict(X_test)
test_accuracy=accuracy_score(Y_test,Y_test_prime)
print(test_accuracy)

#Predictive System
input_data=(7.3,0.65,0.0,1.2,0.065,15.0,21.0,0.9946,3.39,0.47,10.0)
input_data=np.asarray(input_data)
input_data_reshaped=input_data.reshape(1,-1)
prediction=model.predict(input_data_reshaped)
print(prediction)





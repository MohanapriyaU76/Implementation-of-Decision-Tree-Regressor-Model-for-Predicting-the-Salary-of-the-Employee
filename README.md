# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.import dataset and get data info

2.check for null values

3.Map values for position column

4.Split the dataset into train and test set

5.Import decision tree regressor and fit it for data

6.Calculate MSE,R2 and y predict.


## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Mohanapriya U
RegisterNumber: 212220040091
*/
```
import pandas as pd

data=pd.read_csv('/content/Salary.csv')

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

data["Position"]=le.fit_transform(data["Position"])

data.head()

x=data[["Position","Level"]]

x.head()

y=data[["Salary"]]

y.head()

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=200)

from sklearn.tree import DecisionTreeRegressor

dt=DecisionTreeRegressor()

dt.fit(x_train,y_train)

y_pred=dt.predict(x_test)

from sklearn import metrics

mse=metrics.mean_squared_error(y_test,y_pred)

mse

r2=metrics.r2_score(y_test,y_pred)

r2

dt.predict([[5,6]])




## Output:
![Decision Tree Regressor Model for Predicting the Salary of the Employee](sam.png)

1.data.head()

![image](https://github.com/MohanapriyaU76/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/133958624/6123bf95-3a7e-4824-96c3-fc0fa748efe0)

2.data.info()

![image](https://github.com/MohanapriyaU76/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/133958624/dd0897fc-48a2-4ec0-9e84-17d0a71c1c5d)

3.isnull() and sum()

![image](https://github.com/MohanapriyaU76/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/133958624/ca77869f-f41d-4c4c-8cc0-08ae5eb605c3)

4.data.head() for salary

![image](https://github.com/MohanapriyaU76/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/133958624/11b91c9e-8286-41bf-b69c-daa734263b84)

5.MSE Value

![image](https://github.com/MohanapriyaU76/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/133958624/cc2a916c-25f0-46a8-b327-12839501b7e0)

6.r2 value

![image](https://github.com/MohanapriyaU76/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/133958624/0723325e-58f1-406b-920b-6c5b0af855d3)

7.data prediction

![image](https://github.com/MohanapriyaU76/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/133958624/a6b8e3d3-676f-4928-8c12-8510f38d4ad3)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.

#HANDS ON
#how diabetes disease develops or worsens
#over time in a patient w.r.t. BML.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#data = sb.load_dataset('diabetes')

diabetes = load_diabetes()
#print(diabetes.info())
data = pd.DataFrame(diabetes.data, columns= diabetes.feature_names)
data['target'] = diabetes.target
#View data structure and summary
print(data.info())

#Normalisation
#Scatter plot BMI vs Target
scalar = MinMaxScaler()
data[['age','bmi','target','s1','s2','s3','s4','s5','s6']]=scalar.fit_transform(data[['age','bmi','target','s1','s2','s3','s4','s5','s6']])

print(data.describe())
plt.figure(figsize=(8,6))
sb.scatterplot(x = 'bmi', y ='target', data = data)
plt.title("Disease fluctuation vs BMI index")
plt.xlabel('bmi index')
plt.ylabel('target')
plt.grid(True)
plt.show()

#Linear Regression Model
x = data[['bmi']]
y = data[['target']]
x_train,x_test, y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)

#predict for a new BMI  value
check_bmi = pd.DataFrame({'bmi':[0.08]})
result = model.predict(check_bmi)
print("Predicted Diseases Progression for BMI 0.08(normalisation):", result)

#plot regression line
plt.scatter(x, y, color = 'blue')
plt.plot(x,model.predict(x), color = 'red', linewidth = 3)
plt.xlabel('BMI (Normalised')
plt.ylabel('Disease progression normalised')
plt.title('Linear Regression Fit')
plt.show()

correlation = data[['bmi', 'target']].corr()
print("Correlation between BMI and disease progression:\n", correlation)

import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

#Loading Iris Dataset From datasets in scikit-learn
iris = datasets.load_iris() 

# Saving data from dataset
iris_X=iris.data

# Declaring Training And Test Data
iris_X_train=iris_X[:]
iris_X_test=iris_X[:]

# Declaring Features Of Training And Test Data
iris_Y_train=iris.target[:]
iris_Y_test=iris.target[:]

# Loading Linear Regression Model
model=linear_model.LinearRegression()

#  This method is used to train the model. During training, the model learns patterns and relationships between input features (iris_X_train) and target values (iris_Y_train) in the training dataset. The specific learning algorithm and optimization procedure depend on the type of model being used.
model.fit(iris_X_train,iris_Y_train)

# Predicting The Features Of Each Element Of Test Data
#iris_Y_predicted=model.predict(iris_X_test)
#print(iris_Y_predicted)
#print(np.round(iris_Y_predicted))

#The mean_squared_error() function in scikit-learn is a metric used to evaluate the performance of a regression model by measuring the average squared difference between the actual target values and the predicted values. 
#print("Mean Squared Error Is : ",mean_squared_error(iris_Y_test,iris_Y_predicted))

# In scikit-learn, when you train a linear regression model using the LinearRegression class, the trained model contains attributes coef_ and intercept_ that represent the coefficients (weights) and intercept of the linear regression equation, respectively. Here's how you can print these attributes:
#print("Weights : ",model.coef_)
#print("Intercept : ",model.intercept_)

print("\nThis project explores the unconventional approach of using regression techniques to classify Iris flowers from the well-known scikit-learn Iris dataset. It aims to predict the flower species (Iris setosa, versicolor, or virginica) based on sepal and petal measurements (length and width) through a regression model.\n")

array_of_inputs=[]
for i in range(4):
    if (i==0):
        feature="Sepal Length (in cm)"
    elif(i==1):
        feature="Sepal Width (in cm)"
    elif(i==2):
        feature="Petal Length (in cm)"
    elif(i==3):
        feature="Sepal Width (in cm)"
        try:
            user_input=float(input(f"Enter {feature} : "))
            array_of_inputs.append(user_input)
            array_of_inputs=np.array(array_of_inputs).reshape(1,-1)  #Converting array_of_inputs from a list to numpy array
            # .reshape(1, -1): This reshapes the array to have one row (1) and an inferred number of columns (-1). Using -1 as one of the dimensions means that NumPy will infer the number of elements in that dimension based on the size of the original array. So, in this case, it will reshape the array to have one row and as many columns as there are elements in the original list.
        except:
            print("\nYou Should Have Typed Float Value Only\n".upper().center(20))
            exit()

            
#print(array_of_inputs)

#print(np.round(model.predict(array_of_inputs)))

if (np.round(model.predict(array_of_inputs))<=0.5):
    print("\nThe Given Flower Is Iris Setosa\n")
if (0.5<np.round(model.predict(array_of_inputs))<1.6):
    print("\nThe Given Flower Is Iris Versicolor\n")
if (np.round(model.predict(array_of_inputs))>=1.6):
    print("\nThe Given Flower Is Iris Virginica\n")
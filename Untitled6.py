#!/usr/bin/env python
# coding: utf-8

# In[1]:


##Lab.1 Write a program to implement k-Nearest Neighbour algorithm to classify the iris data set Print both correct and wrong predictions.

from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import datasets
iris=datasets.load_iris() 
print("Iris Data set loaded...")
x_train, x_test, y_train, y_test = train_test_split(iris.data,iris.target,test_size=0.1)
#random_state=0
for i in range(len(iris.target_names)):
    print("Label", i , "-",str(iris.target_names[i]))
classifier = KNeighborsClassifier(n_neighbors=2)
classifier.fit(x_train, y_train)
y_pred=classifier.predict(x_test)
print("Results of Classification using K-nn with K=1 ") 
for r in range(0,len(x_test)):
    print(" Sample:", str(x_test[r]), " Actual-label:", str(y_test[r])," Predicted-label:", str(y_pred[r]))

    print("Classification Accuracy :" , classifier.score(x_test,y_test));


# In[12]:


# Lab.2 Develop a program to apply K-means algorithm to cluster a set of data stored in .CSV file. Use the same data set for clustering using EM algorithm. Compare the results of 

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# Load data from CSV
data = pd.read_csv("C:\\Users\\Ravi\\Downloads\\iris.csv")

# Select features for clustering
features = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

# Apply K-means clustering
kmeans = KMeans(n_clusters=3)  # Adjust the number of clusters as needed
kmeans_labels = kmeans.fit_predict(features)

# Apply EM algorithm (Gaussian Mixture Model)
em = GaussianMixture(n_components=3)  # Adjust the number of clusters as needed
em_labels = em.fit_predict(features)

# Visualize clustering results
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(data['SepalLengthCm'], data['SepalWidthCm'], c=kmeans_labels, cmap='viridis', edgecolor='k')
plt.title('K-means Clustering')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.scatter(data['SepalLengthCm'], data['SepalWidthCm'], c=em_labels, cmap='viridis', edgecolor='k')
plt.title('EM Clustering')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.colorbar()

plt.tight_layout()
plt.show()

# Comment on the quality of clustering
# Calculate inertia (for K-means) as an indicator of clustering quality
kmeans_inertia = kmeans.inertia_
print(f"K-means inertia: {kmeans_inertia}")

 


# In[13]:


# lab.3 Implement the non-parametric Locally Weighted Regressionalgorithm in order to fit data points. Select appropriate data set for your experiment and draw graphs 

import numpy as np
import matplotlib.pyplot as plt

def local_regression(x0, X, Y, tau):
    x0 = [1, x0]   
    X = [[1, i] for i in X]
    X = np.asarray(X)
    xw = (X.T) * np.exp(np.sum((X - x0) ** 2, axis=1) / (-2 * tau))
    beta = np.linalg.pinv(xw @ X) @ xw @ Y @ x0  
    return beta    

def draw(tau):
    prediction = [local_regression(x0, X, Y, tau) for x0 in domain]
    plt.plot(X, Y, 'o', color='black')
    plt.plot(domain, prediction, color='red')
    plt.show()

X = np.linspace(-3, 3, num=1000)
domain = X
Y = np.log(np.abs(X ** 2 - 1) + .5)

draw(10)
draw(0.1)
draw(0.01)
draw(0.001)


# In[14]:


# Lab 4. Build an Artificial Neural Network by implementing the Backpropagation algorithm and test the same using appropriate data sets

import numpy as np
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)
X = X/np.amax(X,axis=0) # maximum of X array longitudinally
y = y/100

#Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)

#Variable initialization
epoch=5000                #Setting training iterations
lr=0.1                    #Setting learning rate
inputlayer_neurons = 2    #number of features in data set
hiddenlayer_neurons = 3   #number of hidden layers neurons
output_neurons = 1        #number of neurons at output layer

#weight and bias initialization
wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh=np.random.uniform(size=(1,hiddenlayer_neurons))
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout=np.random.uniform(size=(1,output_neurons))


#draws a random range of numbers uniformly of dim x*y
for i in range(epoch):
    
#Forward Propogation
    hinp1=np.dot(X,wh)
    hinp=hinp1 + bh
    hlayer_act = sigmoid(hinp)
    outinp1=np.dot(hlayer_act,wout)
    outinp= outinp1+ bout
    output = sigmoid(outinp)
    
#Backpropagation
    EO = y-output
    outgrad = derivatives_sigmoid(output)
    d_output = EO* outgrad
    EH = d_output.dot(wout.T)

#how much hidden layer wts contributed to error
    hiddengrad = derivatives_sigmoid(hlayer_act)
    d_hiddenlayer = EH * hiddengrad
    
# dotproduct of nextlayererror and currentlayerop
    wout += hlayer_act.T.dot(d_output) *lr
    wh += X.T.dot(d_hiddenlayer) *lr
    
    
print("Input: \n" + str(X))
print("Actual Output: \n" + str(y))
print("Predicted Output: \n" ,output)


# In[16]:


# Lab 5.Demonstrate Genetic algorithm by taking a suitable data for any simple application

import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms  # Add 'algorithms' import here

# Create city coordinates (for demonstration purposes, generate random coordinates)
num_cities = 10
city_coords = np.array([[random.random(), random.random()] for _ in range(num_cities)])

# Define the evaluation function for the TSP
def tsp_distance(individual):
    distance = 0
    for i in range(len(individual)):
        city_start = individual[i]
        city_end = individual[(i + 1) % len(individual)]
        distance += np.linalg.norm(city_coords[city_start] - city_coords[city_end])
    return distance,

# Genetic Algorithm parameters
population_size = 100
num_generations = 50
mutation_rate = 0.1
crossover_rate = 0.8

# Create a fitness class maximizing 1 / distance
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(num_cities), num_cities)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", tsp_distance)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=mutation_rate)
toolbox.register("select", tools.selTournament, tournsize=3)

population = toolbox.population(n=population_size)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("min", np.min)
stats.register("avg", np.mean)

best = tools.HallOfFame(1)

# Perform the evolution using DEAP's algorithms
population, logbook = algorithms.eaSimple(population, toolbox, cxpb=crossover_rate, mutpb=mutation_rate,
                                          ngen=num_generations, stats=stats, halloffame=best, verbose=True)

best_route = best[0]
best_distance = tsp_distance(best_route)[0]

print("Best Route:", best_route)
print("Best Distance:", best_distance)

# Plotting the best route
plt.figure(figsize=(6, 6))
plt.scatter(city_coords[:, 0], city_coords[:, 1], c='red', label='Cities')
best_route_coords = city_coords[best_route]
plt.plot(best_route_coords[:, 0], best_route_coords[:, 1], linestyle='-', marker='o', color='blue')
plt.title('Best Route for TSP')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.legend()
plt.show()
 


# In[17]:


# Lab 6. Demonstrate Q learning algorithm with suitable assumption for a problem statement

import numpy as np
import random

# Setting up the Environment
# Defining the Reward Matrix
R = np.array([[-1, -1, -1, -1, 0, -1],
              [-1, -1, -1, 0, -1, 100],
              [-1, -1, -1, 0, -1, -1],
              [-1, 0, 0, -1, 0, -1],
              [0, -1, -1, 0, -1, 100],
              [-1, 0, -1, -1, 0, 100]])

# Defining the Q-Table
Q = np.zeros_like(R)

# Defining the Hyperparameters
gamma = 0.8
alpha = 0.1
epsilon = 0.1

# Training the Agent
for episode in range(1000):
    state = random.randint(0, 5)
    while state != 5:
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, 5)
        else:
            action = np.argmax(Q[state])
        next_state = action
        reward = R[state, action]
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
        state = next_state

# Testing the Agent
state = 2
steps = [state]
while state != 5:
    action = np.argmax(Q[state])
    state = action
    steps.append(state)

# Printing the Optimal Path
print("Optimal Path: ", steps)


# In[ ]:





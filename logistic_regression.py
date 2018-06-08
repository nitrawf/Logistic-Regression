import numpy as np
import matplotlib.pyplot as plt
import csv

def loaddata():
    with open("set.csv",newline='') as csvfile:
        lines=csv.reader(csvfile,delimiter=",",quoting=csv.QUOTE_NONE, quotechar='')
        for x in lines:
            templist=[]
            for i in range(len(x)-1):
                if x[i]=="Female":
                    x[i]=0
                elif x[i]=="Male":
                    x[i]=1
                templist.append(float(x[i]))
            X_list.append(templist)
            if x[len(x)-1]=="1":
                Y_list.append([0])
            else:
                Y_list.append([1])

def sigmoid(z):
    s=1/(1+np.exp(-z))
    return s

def initialize(dim):
    w=np.zeros((dim,1),dtype=float)
    b=0
    return w,b

def propagate(w, b, X, Y):        
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X)+b)                                 
    cost = (-1/m)*np.sum(np.dot(Y,np.log(A).T)+np.dot((1-Y),np.log(1-A).T))              
    dw = (1/m)*np.dot(X,(A-Y).T)
    db = (1/m)*np.sum(A-Y)    
    cost = np.squeeze(cost)  
    grads = {"dw": dw,
             "db": db}    
    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):     
    costs = []      
    for i in range(num_iterations):
        grads, cost = propagate(w,b,X,Y)
        dw = grads["dw"]
        db = grads["db"]      
        w = w-learning_rate*dw
        b = b-learning_rate*db
        if i%100==0:
            costs.append(cost)       
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))   
    params = {"w": w,
              "b": b}
    grads = {"dw": dw,
             "db": db}    
    return params, grads, costs


def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)        
    A = sigmoid(np.dot(w.T,X)+b)
    for i in range(A.shape[1]):
        if A[0][i]<=0.75:
            Y_prediction[0][i]=0
        else:
            Y_prediction[0][i]=1   
    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):  
    w, b = initialize(X_train.shape[0])
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)        
    w = parameters["w"]
    b = parameters["b"]    
    Y_prediction_test = predict(w,b,X_test)
    Y_prediction_train = predict(w,b,X_train)   
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100)) 
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}   
    return d  

       

X_list=[]
Y_list=[]                
loaddata()
X=np.array(X_list).T
Y=np.array(Y_list).T
Xmeans=np.mean(X,axis=1).reshape(X.shape[0],1)
Xdevs=np.std(X,axis=1).reshape(X.shape[0],1)
X=(X-Xmeans)/Xdevs
X_trainset=X[:,0:400]
Y_trainset=Y[:,0:400]
X_testset=X[:,400:]
Y_testset=Y[:,400:]    
d = model(X_trainset, Y_trainset, X_testset, Y_testset, num_iterations = 2000, learning_rate = 0.005, print_cost = True)
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()


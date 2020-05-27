import pandas as pd
import string
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
import warnings
import math
warnings.filterwarnings("ignore")

def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s

def relu(x):
    s = np.maximum(0,x)
    return s

def linear_relu(W, X, b):
    z = np.dot(W, X) + b
    a = relu(z)
    return z, a

def linear_tanh(W, X, b):
    z = np.dot(W, X) + b
    a = np.tanh(z)
    return z, a

def linear_sigmoid(W, X, b):
    z = np.dot(W, X) + b
    a = sigmoid(z)
    return z, a

def initialize_parameters(layer_dims):
    parameters = {}
    L = len(layer_dims) # number of layers in the network
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*  np.sqrt(2 / layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        assert parameters['W' + str(l)].shape[0] == layer_dims[l], layer_dims[l-1]
        assert parameters['W' + str(l)].shape[0] == layer_dims[l], 1
    return parameters

def forward_propagation(X, Y, parameters, lambd = 0):
    L = len(parameters) // 2 # number of layers in the neural networks
    cache = {}
    cache['a0'] = X
    for l in range(L-1):
        zkey = 'z' + str(l+1)
        akey = 'a' + str(l+1)
        Wkey = 'W' + str(l+1)
        bkey = 'b' + str(l+1)
        cache[Wkey] = parameters[Wkey]
        cache[bkey] = parameters[bkey]
        cache[zkey], cache[akey] = linear_tanh(cache[Wkey], cache['a' + str(l)], cache[bkey])
    zkey = 'z' + str(L)
    akey = 'a' + str(L)
    Wkey = 'W' + str(L)
    bkey = 'b' + str(L)
    cache[Wkey] = parameters[Wkey]
    cache[bkey] = parameters[bkey]
    cache[zkey], cache[akey] = linear_sigmoid(cache[Wkey], cache['a' + str(L-1)], cache[bkey])
    if lambd > 0:
        cost = compute_cost_with_regularization(cache[akey], Y, parameters, lambd)
    else:
        cost = compute_cost(cache[akey], Y)
    return cost, cache

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (this turns [[17]] into 17).
    assert(cost.shape == ())
    return cost

def compute_cost_with_regularization(AL, Y, parameters, lambd):
    m = Y.shape[1]
    L = len(parameters) // 2 # number of layers in the neural networks
    tmp = 0
    for l in range(1, L+1):
        tmp += np.sum(np.square(parameters['W' + str(l)]))
    cross_entropy_cost = compute_cost(AL, Y)
    L2_regularization_cost = (1/m) * (lambd/2) * tmp
    cost = cross_entropy_cost + L2_regularization_cost
    return cost

def predict(X, y, parameters):
    m = X.shape[1]
    p = np.zeros((1,m), dtype = np.int)
    _, cache = forward_propagation(X, y, parameters)
    L = len(parameters) // 2 # number of layers in the neural networks
    al = cache['a' + str(L)]
    for i in range(0, al.shape[1]):
        if al[0,i] > 0.45:
            p[0,i] = 1
        else:
            p[0,i] = 0
    # print results
    #print ("predictions: " + str(p[0,:]))
    #print ("true labels: " + str(y[0,:]))
    print("Accuracy: "  + str(np.mean((p[0,:] == y[0,:]))))
    return p

def predict_t(X, parameters):
    m = X.shape[1]
    p = np.zeros((1,m), dtype = np.int)
    y = np.zeros((1,m), dtype = np.int)
    _, cache = forward_propagation(X, y, parameters)
    L = len(parameters) // 2 # number of layers in the neural networks
    al = cache['a' + str(L)]
    for i in range(0, al.shape[1]):
        if al[0,i] > 0.45:
            p[0,i] = 1
        else:
            p[0,i] = 0         
    return p

def update_parameters_with_gd(parameters, grads, learning_rate):
    L = len(parameters) // 2 # number of layers in the neural networks
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - (learning_rate * grads['dW' + str(l +1)])
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - (learning_rate * grads['db' + str(l +1)])
    return parameters

def initialize_velocity(parameters):
    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}
    for l in range(L):
        v["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)  
    return v

def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    L = len(parameters) // 2 # number of layers in the neural networks
    for l in range(L):
        v["dW" + str(l+1)] = beta * v["dW" + str(l+1)] + (1 - beta) * grads["dW" + str(l+1)]
        v["db" + str(l+1)] = beta * v["db" + str(l+1)] + (1 - beta) * grads["db" + str(l+1)]
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * v["db" + str(l+1)]
    return parameters, v

def initialize_adam(parameters) :
    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}
    s = {}
    for l in range(L):
        v["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
        s["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        s["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
    return v, s

def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    
    L = len(parameters) // 2                 # number of layers in the neural networks
    v_corrected = {}                         # Initializing first moment estimate, python dictionary
    s_corrected = {}                         # Initializing second moment estimate, python dictionary
    for l in range(L):
        v["dW" + str(l+1)] = beta1 * v["dW" + str(l+1)] + (1 - beta1) * grads["dW" + str(l+1)]
        v["db" + str(l+1)] = beta1 * v["db" + str(l+1)] + (1 - beta1) * grads["db" + str(l+1)]
        v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)] / (1 - beta1 ** t)
        v_corrected["db" + str(l+1)] = v["db" + str(l+1)] / (1 - beta1 ** t)
        s["dW" + str(l+1)] = beta2 * s["dW" + str(l+1)] + (1 - beta2) * (grads["dW" + str(l+1)]**2)
        s["db" + str(l+1)] = beta2 * s["db" + str(l+1)] + (1 - beta2) * (grads["db" + str(l+1)]**2)
        s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)] / (1 - beta2 ** t)
        s_corrected["db" + str(l+1)] = s["db" + str(l+1)] / (1 - beta2 ** t)
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * (v_corrected["dW" + str(l+1)] / (np.sqrt(s_corrected["dW" + str(l+1)] ) + epsilon))
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * (v_corrected["db" + str(l+1)] / (np.sqrt(s_corrected["db" + str(l+1)] ) + epsilon))
    return parameters, v, s

def gradient_check(parameters, gradients, X, Y, epsilon = 1e-7, lambd = 0):
    parameters_values, _ = dictionary_to_vector(parameters)
    grad = gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))
    for i in range(num_parameters):
        if lambd > 0:
            thetaplus = np.copy(parameters_values) # Step 1
            thetaplus[i][0] = thetaplus[i][0] + epsilon # Step 2
            J_plus[i], _ = forward_propagation(X, Y, vector_to_dictionary(thetaplus), lambd)# Step 3
            thetaminus = np.copy(parameters_values)               # Step 1
            thetaminus[i][0] = thetaminus[i][0] - epsilon         # Step 2        
            J_minus[i], _ = forward_propagation(X, Y, vector_to_dictionary(thetaminus), lambd)# Step 3
        else:
            thetaplus = np.copy(parameters_values) # Step 1
            thetaplus[i][0] = thetaplus[i][0] + epsilon # Step 2
            J_plus[i], _ = forward_propagation(X, Y, vector_to_dictionary(thetaplus))# Step 3
            thetaminus = np.copy(parameters_values)               # Step 1
            thetaminus[i][0] = thetaminus[i][0] - epsilon         # Step 2        
            J_minus[i], _ = forward_propagation(X, Y, vector_to_dictionary(thetaminus))# Step 3
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)
    numerator = np.linalg.norm(grad - gradapprox)                  # Step 1'
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)# Step 2'
    difference = numerator / denominator                           # Step 3'
    if difference > 2e-7:
        print ("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
    else:
        print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")
    return difference

def dictionary_to_vector(parameters):
    keys = []
    count = 0
    for key in ["W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4", "W5", "b5", "W6", "b6"]:
        new_vector = np.reshape(parameters[key], (-1,1))
        keys = keys + [key]*new_vector.shape[0]
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1
    return theta, keys

def vector_to_dictionary(theta):
    parameters = {}
    parameters["W1"] = theta[:384].reshape((24,16))
    parameters["b1"] = theta[384:408].reshape((24,1))
    parameters["W2"] = theta[408:792].reshape((16,24))
    parameters["b2"] = theta[792:808].reshape((16,1))
    parameters["W3"] = theta[808:936].reshape((8,16))
    parameters["b3"] = theta[936:944].reshape((8,1))
    parameters["W4"] = theta[944:976].reshape((4,8))
    parameters["b4"] = theta[976:980].reshape((4,1))
    parameters["W5"] = theta[980:988].reshape((2,4))
    parameters["b5"] = theta[988:990].reshape((2,1))
    parameters["W6"] = theta[990:992].reshape((1,2))
    parameters["b6"] = theta[992:993].reshape((1,1))
    return parameters

def gradients_to_vector(gradients):
    count = 0
    for key in ["dW1", "db1", "dW2", "db2", "dW3", "db3", "dW4", "db4", "dW5", "db5", "dW6", "db6"]:
        new_vector = np.reshape(gradients[key], (-1,1))
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1
    return theta

def gradient_check_test_case(): 
    np.random.seed(1)
    x = np.random.randn(16,16)
    y = np.array([[1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1]])
    W1 = np.random.randn(24,16) 
    b1 = np.random.randn(24,1) 
    W2 = np.random.randn(16,24) 
    b2 = np.random.randn(16,1)
    W3 = np.random.randn(8,16) 
    b3 = np.random.randn(8,1)
    W4 = np.random.randn(4,8) 
    b4 = np.random.randn(4,1) 
    W5 = np.random.randn(2,4) 
    b5 = np.random.randn(2,1) 
    W6 = np.random.randn(1,2) 
    b6 = np.random.randn(1,1) 
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4": b4,
                  "W5": W5,
                  "b5": b5,
                  "W6": W6,
                  "b6": b6}
    return x, y, parameters

def forward_propagation_test_case():
    np.random.seed(1)
    X_assess = np.random.randn(2, 3)
    Y = np.zeros((1, 3))
    b1 = np.random.randn(4,1)
    b2 = np.array([[ -1.3]])
    parameters = {'W1': np.array([[-0.00416758, -0.00056267],
        [-0.02136196,  0.01640271],
        [-0.01793436, -0.00841747],
        [ 0.00502881, -0.01245288]]),
     'W2': np.array([[-0.01057952, -0.00909008,  0.00551454,  0.02292208]]),
     'b1': b1,
     'b2': b2}
    return X_assess, Y, parameters

def compute_cost_with_regularization_test_case():
    np.random.seed(1)
    Y_assess = np.array([[1, 1, 0, 1, 0]])
    W1 = np.random.randn(2, 3)
    b1 = np.random.randn(2, 1)
    W2 = np.random.randn(3, 2)
    b2 = np.random.randn(3, 1)
    W3 = np.random.randn(1, 3)
    b3 = np.random.randn(1, 1)
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
    a3 = np.array([[ 0.40682402,  0.01629284,  0.16722898,  0.10118111,  0.40682402]])
    return a3, Y_assess, parameters
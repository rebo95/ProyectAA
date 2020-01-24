import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D 
import math
from math import e
from pandas.io.parsers import read_csv
from sklearn.preprocessing import PolynomialFeatures

import scipy.optimize as opt
import random

from sklearn import svm 

#_______________________________________________________________________________________

def data_csv(file_name):
    "Takes the data from the csv file and tranfers it to a numpy array"

    values_ = read_csv(file_name, header=None).values

    return values_.astype(float)


def data_builder(data):

    X = data[:, 2:]
    Y = data[:, 1]

    return X, Y

#_______________________________________________________________________________________

def data_visualization(X, Y):

    pos_0 = np.where(Y == 0)
    pos_1 = np.where(Y == 1)

    plt.xlabel('Radius')
    plt.ylabel('Perimeter')
    plt.scatter(X[pos_0, 0], X[pos_0, 3], marker = '+', c = "green")
    plt.scatter(X[pos_1, 0], X[pos_1, 3],  marker = '.',c = 'red')

    plt.show()



def vectors_coincidence_percentage(a, b):
    #Calcula el porcentaje de coincidencia dados dos vectores a, b
    coincidences_array = a == b

    coincidences = sum(map(lambda coincidences_array : coincidences_array == True, coincidences_array  ))
    percentage =100 * coincidences/coincidences_array.shape

    return percentage


#_______________________________________________________________________________________
#Regresión lineal, gradiente descendiente.
#_______________________________________________________________________________________

#______________________
#Funcion de normalizado
#______________________
def normalize(X):

    X_normalized = np.zeros((X.shape[0], X.shape[1]))

    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])
    
    np.mean(X, axis = 0, out = mu)
    np.std(X, axis = 0, out = sigma)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X_normalized[i,j] = (X[i,j] - mu[j])/sigma[j] 
    
    return X_normalized, mu, sigma


#__________________
#Funcion hipótesis
#__________________
def H_Theta(X, Z): #Hipótesis del mocelo lineal vectorizada 
    return np.dot(X, Z)

#__________________________________
#Funcion de costes regresión lineal
#__________________________________
def cost_function_linear_regression(X, Y, Theta): #funcion de costes vectorizada
    H = H_Theta(X,Theta)
    Aux = (H-Y)**2
    sumatory = Aux.sum()/(2 * len(X))
    return sumatory


#_____________________________________________
#Funcion normal, ajuste de la regresión lineal
#_____________________________________________

def normalEcuation(X,Y): #nos da los valores de theta que mejor se ajustan a nuestra regresión lineal

    #theta = (XT * X)**-1 * XT * Y 

    XT = np.transpose(X)

    XT__X = np.matmul(XT, X)

    XT__X_Inv = np.linalg.pinv(XT__X, rcond = 10**(-15))

    XT__X_Inv__XT = np.matmul(XT__X_Inv, XT)

    thetas = np.matmul(XT__X_Inv__XT, Y)

    return thetas

#_____________________________
#Funcion descenso de gradiente
#_____________________________
def gradient_descent(X, Y, alpha):
    
    m = X.shape[0]

    #construimos matriz Z
    th0 = 0.0
    th1 = 0.0
    th_n = 0.0

    Z = np.zeros(X.shape[1]) #estamos tomando la dimensión de la X con la columna de 1, de tal manera que esta si coincide con el valor de ladimension que tiene que tener el vector de thetas

    Z_ = np.zeros(X.shape[1] - 1)

    alpha_m = (alpha/m)

    Thetas = np.array([Z]) #almacena los thetas que forman parte de la hipotesis h_theta
    costs = np.array([]) #almacena los costes obtenidos durante el descenso de gradiente
 
    for i in range(1500):

        #Calculo de Theta 0
        #Sumatorio para el calculo de Theta0
        sum1 = H_Theta(X, Z) - Y
        sum1_ = sum1.sum()
        th0 -= alpha_m * sum1_

        #Calculo Theta 1, 2, 3 ... n
        #Sumatorio para el calculo de Thetan
        for k in range(X.shape[1] - 1):
            sum2 =  (H_Theta(X, Z) - Y) * X[:, k + 1]
            sum2_ = sum2.sum()
            th_n -= alpha_m * sum2_ #vamos calculando cada uno de los thn
            Z_[k] = th_n #almacenamos los thn calculados en un vector provisional


        #Actualizamos los nuevos thetas del vector Z    
        Z[0] = th0
    
        for p in range(X.shape[1]-1):
            Z[p+1] = Z_[p]

        Thetas = np.append(Thetas, [Z], axis= 0)

        #funcion de costes
        J = cost_function_linear_regression(X,Y, Z)

        costs = np.append(costs, [J], axis = 0)

    return Thetas, costs

#_______________________________________
#Main Regresion Lineal, varias variables
#________________________________________
def regresion_lineal_multiples_variables():
    
    cancer_data = data_csv("data.csv")
    X, Y = data_builder(cancer_data)
    
    
    X_normalizada, mu, sigma = normalize(X)


    X = np.hstack([np.ones([np.shape(X)[0], 1]), X])
    X_shape_1 = np.shape(X_normalizada)[0]


    X_normalizada = np.hstack([np.ones([X_shape_1, 1]), X_normalizada]) #le añadimos la columna de unos a la matriz ya normalizada

    Thetas, costs = gradient_descent(X_normalizada, Y, 0.0022) #los valores de theta aquí son los obtenidos normalizando la matriz, esto es, necesitamos "desnormalizarlos"
    Thetas_normal_Ecuation = normalEcuation(X, Y)


    normal_prediction, gradient_descent_prediction = prediction_vectors(X, Thetas_normal_Ecuation, Thetas, mu, sigma)

    normal_prediction = vector_prediction_moddel(normal_prediction)
    gradient_descent_prediction = vector_prediction_moddel(gradient_descent_prediction)

    print(vectors_coincidence_percentage(normal_prediction, gradient_descent_prediction))#100 porciento de coincidencia con gradiente descendiente en comparación con distribución normal

    print(vectors_coincidence_percentage(gradient_descent_prediction, Y)) #86.99 porciento de acierto el la predicción con gradiente descendiente


#___________________________________________________________________
#Cálculo vectores prediccion ecuación normal y descenso de gradiente
#___________________________________________________________________
def prediction_vectors(X, Thetas_normal_Ecuation, Thetas, mu, sigma):

    shape_thetas = np.shape(Thetas)[0]-1
    cancer_samples = X.shape[0]

    normal_predictions = np.zeros(cancer_samples)
    gradient_descendant_predictions = normal_predictions

    for i in range(cancer_samples):

        cancer_values = X[i, :]
        cancer_Not_normalize_values = convert_to_list(cancer_values)
        prediccion_normal_ecuation = H_Theta(cancer_Not_normalize_values ,Thetas_normal_Ecuation)

        normal_predictions[i] = prediccion_normal_ecuation

        cancer_normalize_values = normalized_test_value(mu, sigma, cancer_Not_normalize_values)
        prediccion_gradiente_descendiente = H_Theta(cancer_normalize_values, Thetas[shape_thetas])

        gradient_descendant_predictions[i] = prediccion_gradiente_descendiente

    return normal_predictions, gradient_descendant_predictions

#______________________________________________________________________________________
#Modelizado vectores predicción, a 0 todo lo que baja de 0.5 y a 1 todo lo que es mayor
#______________________________________________________________________________________
def vector_prediction_moddel(vector):

    moddel_vector = np.zeros(vector.shape[0])
    moddel_vector = (vector >= 0.5).astype(np.int)
    
    return moddel_vector


#metodo auxiliar: convierte cada una de las filas o en nuestro caso caso de análisis en una lista para poder tratarlo posteriormente
def normalized_test_value(mu, sigma, row_list) :

    dim = mu.shape[0]
    normalized_values = []
    normalized_values.append(1)

    for i in range(1, dim +1):
        normalized_values.append((row_list[0][i] - mu[i-1])/sigma[i-1])

    normalized_values = [normalized_values]
    return normalized_values


#metodo auxiliar: convierte un vector a una lista
def convert_to_list(row):

    values_list = []
    for i in range(row.shape[0]):
        values_list.append(row[i])

    values_list = [values_list]

    return values_list


#FIN DESCENSO DE GRADIENTE
#----------------------------------------------------------------------------------------------------------------------------------------------------------


#=======================================================================================
#Regresión logística
#=======================================================================================
#____________________
#Función sigmoide
#____________________
def sigmoide_function(X):

    e_z = 1 / np.power(math.e, X) #np.power => First array elements raised to powers from second array, element-wise.

    sigmoide = 1/(1 + e_z)

    return sigmoide

#______________________
#Función hipotesis nula
#______________________
def H(X, Z): #Hipótesis del mocelo lineal vectorizada 
    return np.dot(X, Z)


#____________________
#Función de coste
#____________________

def cost(Thetas, X, Y):

    m = X.shape[0] 
    #J(θ) = −(1/m) * (A + B * C)
    #J(θ) = −(1/m) * ((log (g(Xθ)))T * y + (log (1 − g(Xθ)))T * (1 − y))

    #A
    X_Teta = np.dot(X, Thetas)
    g_X_Thetas = sigmoide_function(X_Teta)
    log_g_X_Thetas = np.log(g_X_Thetas)
    T_log_g_X_Thetas = np.transpose(log_g_X_Thetas)
    y_T_log_g_X_Thetas = np.dot(T_log_g_X_Thetas, Y)

    A = y_T_log_g_X_Thetas

    #B
    one_g_X_Thetas = 1 - g_X_Thetas
    log_one_g_X_Thetas = np.log(one_g_X_Thetas)
    T_log_one_g_X_Thetas = np.transpose(log_one_g_X_Thetas)

    B = T_log_one_g_X_Thetas

    #C
    C = 1 - Y

    J = (-1/m) * (A + (np.dot(B, C)))

    

    _gradient = gradient(Thetas, X, Y)


    return J, _gradient

def two_power(X):
    return X**2

#_____________________________
#Función de coste regularizado
#_____________________________

def cost_regularized(Thetas, X, Y, h):

    m = X.shape[0] 
    Thetas_ = Thetas

    #J(θ) = (cost(Thetas, X, Y)) + D
    #J(θ) = [−(1/m) * ((log (g(Xθ)))T * y + (log (1 − g(Xθ)))T * (1 − y)))] + (λ/2m)*E(Theta^2)

    cost_ = cost(Thetas, X, Y)[0]
   
    #D
    Thetas_ = two_power(Thetas_)

    D = h/(2*m) * np.sum(Thetas_)

    J_regularized = (cost_) + D
    
    g = gradient_regularized(Thetas, X, Y, h)

    return J_regularized, g

#____________________
#Función de gradiente 
#____________________

def gradient(Thetas, X, Y):

    m = X.shape[0]
    #(δJ(θ)/δθj) =(1/m)*XT*(g(Xθ) − y)
    

    X_Teta = np.dot(X, Thetas)
    g_X_Thetas = sigmoide_function(X_Teta)

    X_T = np.transpose(X)

    gradient = (1/m)*(np.dot(X_T, g_X_Thetas - Y ))

    return gradient

#_________________________________
#Función de gradiente regularizado
#_________________________________

def gradient_regularized(Thetas, X, Y, h):

    m = X.shape[0]
    #(δJ(θ)/δθj) =(1/m)*XT*(g(Xθ) − y) + (λ/2m)(Theta)
    gradient_ = gradient(Thetas, X, Y)

    g_regularized = gradient_ + (h/m)*Thetas
    
    return g_regularized

#_________________________________
#Función thetas optimos
#_________________________________
def optimized_parameters(Thetas, X, Y):

    result = opt.fmin_tnc(func = cost, x0 = Thetas, fprime = gradient, args = (X, Y) )
    theta_opt = result[0]

    return theta_opt
    
def minimice_rl(cost, Thetas, X, Y):
    #Calcula los parámetros optimos de pesos para nuestra red neuronal
    fmin = opt.minimize(fun=cost, x0=Thetas, args=(X, Y), 
    method='Nelder-Mead', jac=True, options={'maxiter': 550})

    result = fmin.x
    return result

def minimice_rl_r(cost, Thetas, X, Y, h, iter):
    #Calcula los parámetros optimos de pesos para nuestra red neuronal
    fmin = opt.minimize(fun=cost_regularized, x0=Thetas, args=(X, Y, h), 
    method='Nelder-Mead', jac=True, options={'maxiter': iter})

    result = fmin.x
    return result

#____________________________________
#Función thetas optimos regularizados
#____________________________________
def optimized_parameters_regularized(Thetas, X, Y, h):

    result = opt.fmin_tnc(func = cost_regularized, x0 = Thetas, fprime = gradient_regularized, args = (X, Y, h) )
    theta_opt = result[0]

    return theta_opt



def logistic_regresion_evaluation(X, Y, Z):

    H_ = H(X, Z)
    H_sigmoid = sigmoide_function(H_)

    H_sigmoid_evaluated = (H_sigmoid >= 0.5).astype(np.float) #every value that keeps the condition will return true. astyping it to int it will turn it into a one value, having an array ready to compare with Y_
    
    #returns an array where each value will be true if the condition is keeped

    return vectors_coincidence_percentage( Y ,H_sigmoid_evaluated)

    
#____________________________________
#Función Regresión logística
#____________________________________
def logistic_regresion():

    cancer_data = data_csv("data.csv")
    X_, Y_ = data_builder(cancer_data)
    

    X_V = X_ #X_ that will be used for use in vectorized methods, with the addition of a collum of ones as the first group of atributes

    X_m = np.shape(X_)[0]

    X_V = np.hstack([np.ones([X_m,1]),X_]) #adding the one collum

    Thetas = np.zeros(X_V.shape[1])

    cost_ = cost(Thetas, X_V, Y_)[0]
    gradient_ = gradient(Thetas, X_V, Y_)

    #testing and comparing if the resoults are correct having in mind the resoults given in the assignment document

    print(cost_)
    print(gradient_)

    optimized_thetas = minimice_rl(cost, Thetas, X_V, Y_)#Optimus cost is obtained by calling the cost ecuation using the optimized thetas

    print("Thetas optimos: ", optimized_thetas)
    cost_ = cost(optimized_thetas, X_V, Y_)[0]
    print("Coste optimo : ", cost_)

    percentage_correct_clasification = logistic_regresion_evaluation(X_V, Y_, optimized_thetas)

    print("Porcentaje de acierto : ", percentage_correct_clasification)

#________________________________________
#Función Regresión logística regularizada
#________________________________________
def regularized_logistic_regresion(h , iter): #90.6% obtenido de coincidencia, a medida que hemos ido incrementando el número de iteraciones hemos ido consiguiendo un mayor porcentaje de coincidencia

    cancer_data = data_csv("data.csv")
    X_, Y_ = data_builder(cancer_data)
    
    #poly = PolynomialFeatures(degree=6)
    X_poly = X_[:h, :]
    Y__ = Y_[:h]
    #theta and alpha inicialization
    Thetas = np.zeros(X_poly.shape[1])

    cost_r = cost_regularized(Thetas, X_poly, Y__, h)[0]
    gradient_r = gradient_regularized(Thetas, X_poly, Y__, 1)

    optimized_thetas_regularized = minimice_rl_r(cost_regularized, Thetas, X_poly, Y__, 1, iter)

    percentage_correct_clasification = logistic_regresion_evaluation(X_[h:, :], Y_[h:], optimized_thetas_regularized)
    #print("Porcentaje de acierto : ", percentage_correct_clasification)
    return percentage_correct_clasification

def triDi_regresion():

    h = np.arange(50, 500, 28)
    i = np.arange(50,500,28)

    X,Y = np.meshgrid(h, i)

    Z = regularized_logistic_regresion(X, Y)


    fig = plt.figure()
    ax = Axes3D(fig)

    surf = ax.plot_surface(X, Y, Z, cmap= cm.coolwarm, linewidths= 0, antialiaseds = False)
    fig.colorbar(surf, shrink = 0.5, aspect = 5)
    plt.xlabel("NumeroValoresDeEntrenaiento")
    plt.ylabel("Iteraciones")
    plt.show()

def regularized_logistic_regresion_preparation(X, Y):
    z = X
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            z[i, j] = regularized_logistic_regresion(X[i,j], Y[i,j])
    return z


def pintaFunciontasaRegresionIteraciones(_from = 70, to = 700, step = 10, Xlabel = "Iteraciones", Ylabel = "PorcentajeDeAcierto", name = "( h = 1)"):

    x = np.arange(_from, to, step)
    s = x.shape[0]
    y = np.zeros(s)


    for i in range(s):
        y[i] = regularized_logistic_regresion(1,x[i])


    plt.plot(x, y, color = "blue")
    plt.scatter(x, y, color = "blue", linewidths= 0.05)

    
    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)

    plt.title(name)
    plt.show()


#pintaFunciontasaRegresionIteraciones()
triDi_regresion()

#FIN Regresión logística
#----------------------------------------------------------------------------------------------------------------------------------------------------------

#=======================================================================================
#Redes Neuronales
#=======================================================================================


#_________________________________
#Función auxiliar Despliega vector
#_________________________________
#Nos permite desplegar en un vector otros dos
def unrollVect(a, b): 
    thetaVec_ = np.concatenate((np.ravel(a), np.ravel(b)))
    return thetaVec_

#_________________________________
#Función auxiliar Despliega vector
#_________________________________
#Nos permite plegar en un vector otros dos
def rollVector(params, num_entradas, num_ocultas, num_etiquetas):
    vector1 = np.matrix(np.reshape(params[:num_ocultas * (num_entradas + 1)], (num_ocultas, (num_entradas + 1))))
    vector2 = np.matrix(np.reshape(params[num_ocultas * (num_entradas + 1):], (num_etiquetas, (num_ocultas + 1))))

    return vector1, vector2

#_________________________
#Función pesos aleatorios
#_________________________
#Nos permite inicializar dos vectores de pesos aleatorios para cada una de las capas de la red neuronal con entrada L_in y salida L_out
def generate_Random_Weights(L_in, L_out):

    e_ini = math.sqrt(6)/math.sqrt(L_in + L_out)

    e_ini= 0.12

    weights = np.zeros((L_out, 1 + L_in))

    for i in range(L_out):
        for j in range(1 + L_in):

            rnd = random.uniform(-e_ini, e_ini)
            weights[i,j] = rnd

    return weights


#_______________________________________
#Funciones sigmoide para la red neuronal
#_______________________________________

def sigmoid(x):
    return 1/(1 + np.exp((-x)))

def sigmoid_Derived(x):
    return x * (1 - x)

def sigmoid_Gradient(z):
    sig_z = sigmoid(z)
    return np.multiply(sig_z, (1 - sig_z))


#______________________________________________________
#Funciones procesado Y, preparación para la red neuroal
#______________________________________________________
#Devuelve la salida en forma de matriz lista para ser utilizada por nuestros métodos de la red neuronal
def y_onehot(y, X, num_etiquetas):

    m = X.shape[0]

    y_onehot = np.zeros((m, num_etiquetas)) 

    for i in range(m):
        y_onehot[i][int(y[i])] = 1
    
    return y_onehot

#_______________________________
#Función optimización de pesos
#_______________________________
#Calcula los parámetros optimos de pesos para nuestra red neuronal
def minimice(backprop, params, num_entradas, num_ocultas, num_etiquetas, X, y, tasa_aprendizaje, iter ):

    fmin = opt.minimize(fun=backprop, x0=params, args=(num_entradas, num_ocultas, num_etiquetas, X, y, tasa_aprendizaje), 
    method='TNC', jac=True, options={'maxiter': iter})

    result = fmin.x
    return result


#_______________________________
#Función pasada hacia adelante
#_______________________________
#Método pasada hacia adelante para la implementación de la red neuronal
#Nos devuelve los parámetros de activación de la red neuronal y el valor h

def forward(X, theta1, theta2):
    
    m = X.shape[0]

    a1 = np.insert(X, 0, values=np.ones(m), axis=1)
    z2 = np.dot(a1, theta1.T)
    a2 = np.hstack([np.ones([m, 1]), sigmoid(z2)])
    z3 = np.dot(a2, theta2.T)
    h = sigmoid(z3)

    return a1, z2, a2, z3, h


#___________________________
#Función Coste Red Neuronal
#___________________________
#Funcion que calcula el coste base(sin regularizar)
def cost_n_r(params, num_entradas, num_ocultas, num_etiquetas, X, y, tasa_aprendizaje):

    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y) 

    theta1 = np.matrix(np.reshape(params[:num_ocultas * (num_entradas + 1)], (num_ocultas, (num_entradas + 1))))
    theta2 = np.matrix(np.reshape(params[num_ocultas * (num_entradas + 1):], (num_etiquetas, (num_ocultas + 1))))

    h = forward(X, theta1, theta2)[4]

    J = (np.multiply(-y, np.log(h)) - np.multiply((1 - y), np.log(1 - h))).sum() / m

    return J, theta1, theta2

#_______________________________________
#Función Coste Regularizado Red Neuronal
#_______________________________________
#Cálculo del coste con el ajuste de regularización

def cost_Regularized_n_r(params, num_entradas, num_ocultas, num_etiquetas, X, y, tasa_aprendizaje):

    m = X.shape[0]

    J_, theta1, theta2 = cost_n_r(params, num_entradas, num_ocultas, num_etiquetas, X, y, tasa_aprendizaje)
    
    J_regularized =  J_ + (float(tasa_aprendizaje) /
            (2 * m)) * (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))
    
    return J_regularized



#_____________________________
#Función Cálculo de gradientes 
#_____________________________
#Calculo de los deltas del gradiente (no regularizado)
def backProp_Deltas(a1, z2, a2, z3, h, theta1, theta2, y, m):

    delta1 = np.zeros(theta1.shape)
    delta2 = np.zeros(theta2.shape)
    
    d3 = h - y

    z2 = np.insert(z2, 0, values=np.ones(1), axis=1)

    d2 = np.multiply((theta2.T * d3.T).T, sigmoid_Gradient(z2))

    delta1 += (d2[:, 1:]).T * a1
    delta2 += d3.T * a2

    delta1 = delta1 / m
    delta2 = delta2 / m

    return delta1, delta2


#___________________________________________
#Función Cálculo de gradientes regularizados
#___________________________________________
def backProp_Deltas_regularized(a1, z2, a2, z3, h, theta1, theta2, y, m, tasa_aprendizaje):

    delta1, delta2 = backProp_Deltas(a1, z2, a2, z3, h, theta1, theta2, y, m)

    delta1[:, 1:] = delta1[:, 1:] + (theta1[:, 1:] * tasa_aprendizaje) / m
    delta2[:, 1:] = delta2[:, 1:] + (theta2[:, 1:] * tasa_aprendizaje) / m

    return delta1, delta2

#________________________
#Función paso hacia atrás
#________________________
#Pasada hacia adelante y hacia atrás en nuestra red neuronal, nos calcula el gradiente y el coste correspondientes a nuestra red neuronal
def backprop(params, num_entradas, num_ocultas, num_etiquetas, X, y, tasa_aprendizaje, regularize = True):

    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    theta1 = np.matrix(np.reshape(params[:num_ocultas * (num_entradas + 1)], (num_ocultas, (num_entradas + 1))))
    theta2 = np.matrix(np.reshape(params[num_ocultas * (num_entradas + 1):], (num_etiquetas, (num_ocultas + 1))))


    a1, z2, a2, z3, h = forward(X, theta1, theta2)

    if regularize:
        delta1, delta2 = backProp_Deltas_regularized(a1, z2, a2, z3, h, theta1, theta2, y, m, tasa_aprendizaje)
    else :
        delta1, delta2 = backProp_Deltas(a1, z2, a2, z3, h, theta1, theta2, y, m)

    J = cost_Regularized_n_r(params, num_entradas, num_ocultas, num_etiquetas, X, y, tasa_aprendizaje)

    grad = unrollVect(delta1, delta2)

    return J, grad


def neuronal_prediction_vector(sigmoids_matrix) :
    #calcula los valores predichos por la red neuronal dada una matriz de sigmoides 
    # Será generada por la funcón forward.

    samples = sigmoids_matrix.shape[0]
    y = np.zeros(samples)
    
    for i in range(samples):
        y[i] = np.argmax(sigmoids_matrix[i, :])

    return y

def neuronal_succes_percentage(X, y, weights1, weights2) :
    #Compara los valores predichos por la red neuronal para unos 
    sigmoids_matrix = forward(X, weights1, weights2)[4]
    y_ = neuronal_prediction_vector(sigmoids_matrix)
    percentage = vectors_coincidence_percentage(y_, y)

    return percentage


#___________________________________________
#Main Red Neurnal, ajuste a 25 capas ocultas
#___________________________________________
def neuronal_red_main(ocultas, num_valores_entrenamiento, iteraciones, tasa_a):
    
    cancer_data = data_csv("data.csv")
    X, y = data_builder(cancer_data)
    
    tasa_aprendizaje = tasa_a
    num_etiquetas = 2 #num_etiquetas = num_salidas. Maligno y Benigno
    num_entradas = 30
    num_ocultas = ocultas

    theta1 = generate_Random_Weights(num_entradas, num_ocultas)
    theta2 = generate_Random_Weights(num_ocultas, num_etiquetas)

    params_rn = unrollVect(theta1, theta2)
    y_ = y_onehot(y, X, num_etiquetas)

    params_optimiced = minimice(backprop, params_rn, num_entradas, num_ocultas, num_etiquetas, X[:num_valores_entrenamiento, :], y_[:num_valores_entrenamiento], tasa_a, iteraciones)

    theta1_optimiced, theta2_optimiced = rollVector(params_optimiced, num_entradas, num_ocultas, num_etiquetas)

    percentage = neuronal_succes_percentage(X[num_valores_entrenamiento:], y[num_valores_entrenamiento:], theta1_optimiced, theta2_optimiced)
    
    #print("Percentage neuronal red : ", percentage) #hemos llegado a obtener hasta un 93.84% de acierto (oscilan entre el 87 y el 93 % de acierto)
    return percentage

#FIN Red Neouronal
#----------------------------------------------------------------------------------------------------------------------------------------------------------
def gaussian_Kernel(X1, X2, sigma):
    Gram = np.zeros((X1.shape[0], X2.shape[0]))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            x1 = x1.ravel()
            x2 = x2.ravel()
            Gram[i, j] = np.exp(-np.sum(np.square(x1 - x2)) / (2 * (sigma**2)))
    return Gram


def SVM_gaussian_training(X, y, c_param, tol, max_i, sigma):

    svm_ = svm.SVC(C = c_param, kernel="precomputed", tol = tol, max_iter = max_i)
    return svm_.fit(gaussian_Kernel(X, X, sigma=sigma), y)


def draw_Non_Linear_KernelFrontier(X, y , model, sigma):
   
    #Datos que conformarán la curva que servirá de frontera
    x1plot = np.linspace(X[:,0].min(), X[:,0].max(), 100).T
    x2plot = np.linspace(X[:,1].min(), X[:,1].max(), 100).T
    X1, X2 = np.meshgrid(x1plot, x2plot)
    vals = np.zeros(X1.shape)
    for i in range(X1.shape[1]):
        this_X = np.column_stack((X1[:, i], X2[:, i]))
        vals[:, i] = model.predict(gaussian_Kernel(this_X, X, sigma))

    #Frontera de separación
    plt.contour(X1, X2, vals, colors="r", linewidths = 0.1 )
    displayData(X, y)
    plt.show()

def optimal_C_sigma_Parameters(X, y_r, Xval, yval, max_i, tool ):
    
    predictions = dict() #almacenaremos la infrmacion relevante en un diccionario
    for C in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]:
        for sigma in[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]:
            model = SVM_gaussian_training(X, y_r, C, tool, max_i, sigma )
            prediction = model.predict(gaussian_Kernel( Xval, X, sigma))
            predictions[(C, sigma)] = np.mean((prediction != yval).astype(int))
            
    
    C, sigma = min(predictions, key=predictions.get)
    return C, sigma

def displayData(X, y):

    pos_0 = np.where(y == 0)
    neg_0 = np.where(y == 1)

    plt.plot(X[:,0][pos_0], X[:,1][pos_0], "yo")
    plt.plot(X[:,0][neg_0], X[:,1][neg_0], "k+")

def part2_main():
    #Parte 1.2
    c_param = 1
    sigma = 0.1
    tool = 1e-3
    iterations = 1000

    cancer_data = data_csv("data.csv")
    X, y = data_builder(cancer_data)


    y_r = np.ravel(y)

    svm_function_n_l = SVM_gaussian_training(X, y_r, c_param, tool, iterations, sigma)

def part3_main(test_values, X, y):
    #Parte 1.3

    tool = 1e-3
    iterations = 100


    y_r = np.ravel(y)

    Xval = X[:test_values, :]
    yval = y[:test_values]
    optC, optSigma = optimal_C_sigma_Parameters(X, y_r, Xval, yval, iterations, tool)

    svm_function_optimal_C_sigma = SVM_gaussian_training(X, y_r, optC, tool, iterations, optSigma)
    prediction = svm_function_optimal_C_sigma.predict(gaussian_Kernel( X, X, optSigma))
    return vectors_coincidence_percentage(prediction, y)



def pintaFuncionValoresDeTest(_from = 10, to = 30, step = 1, Xlabel = "TamañoValoresDeEntrenamiento", Ylabel = "Porcentaje", name = "( tool = 1e-1  /it = 100 )"):


    cancer_data = data_csv("data.csv")
    _X, _y = data_builder(cancer_data)

    x = np.arange(_from, to, step)
    s = x.shape[0]
    y = np.zeros(s)


    for i in range(s):
        y[i] = part3_main(x[i], _X, _y)

    plt.plot(x, y, color = "blue")
    plt.scatter(x, y, color = "blue", linewidths= 0.05)

    
    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)

    plt.title(name)
    plt.show()



def main():


    #Los campos sobre los que vamos a realizar nuestro análisis y que se corresponden con las columnas de nuestra matriz X son :
    #"radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
    # "compactness_mean","concavity_mean","concave points_mean","symmetry_mean",
    # "fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se",
    # "smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se",
    # "fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst",
    # "smoothness_worst","compactness_worst","concavity_worst","concave points_worst",
    # "symmetry_worst","fractal_dimension_worst",

    #Volcado de datos en una matriz de parámetros y en un vector de resultados.
    #Por comodidad a la hora de la lectura de datos desde el cvs se han modifcado los valores de M (Maligno) y B(Benigno) 
    #por valores numéricos 1(Maligno) y 0(Benigno)
    cancer_data = data_csv("data.csv")
    X, y = data_builder(cancer_data)

    #tenemos una matriz de 569 x 30 en el que cada fila se correspoende con un ejemplo de entrenamiento que se corresponde con una clasificación de cancer

    print(X.shape)

    data_visualization(X, y)



#def neuronal_red_main(ocultas, num_valores_entrenamiento, iteraciones, tasa_a):

def pintaFuncionCapasOcultasDependiente(_from = 3, to = 25, step = 1, Xlabel = "CapasOcultas", Ylabel = "Porcentaje", name = "( v_e = 400  /it = 700 / t_a = 0.02)"):

    x = np.arange(_from, to, step)
    s = x.shape[0]
    y = np.zeros(s)


    for i in range(s):
        y[i] = neuronal_red_main(x[i], 500, 700, 0.02)

    plt.plot(x, y, color = "green")
    plt.scatter(x, y, color = "green", linewidths= 0.05)

    
    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)

    plt.title(name)
    plt.show()


#def neuronal_red_main(ocultas, num_valores_entrenamiento, iteraciones, tasa_a):
def pintaFuncionIteracionesDependiente(_from = 10, to = 700, step = 10, Xlabel = "NumeroDeIteraciones", Ylabel = "PorcentajeDeAcierto", name = "( v_e = 400  / c_o = 8 / t_a = 0.02)"):

    x = np.arange(_from, to, step)
    s = x.shape[0]
    y = np.zeros(s)


    for i in range(s):
        y[i] = neuronal_red_main(8, 500, x[i], 0.02)

    plt.plot(x, y, color = "green")
    plt.scatter(x, y, color = "green", linewidths= 0.05)

    
    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)

    plt.title(name)
    plt.show()

def pintaFunciontasaDeAprendizajeDependiente(_from = 0.05, to = 10, step = 0.05, Xlabel = "TasaDeaprendizaje", Ylabel = "PorcentajeDeAcierto", name = "( v_e = 400  / c_o = 8 / it = 700)"):

    x = np.arange(_from, to, step)
    s = x.shape[0]
    y = np.zeros(s)


    for i in range(s):
        y[i] = neuronal_red_main(8, 500, 700, x[i])


    plt.plot(x, y, color = "green")
    plt.scatter(x, y, color = "green", linewidths= 0.05)

    
    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)

    plt.title(name)
    plt.show()


def pintaFuncion(_from, to, step, function, Xlabel, Ylabel, name):

    x = np.arange(_from, to, step)
    s = x.shape[0]
    y = np.zeros(s)


    for i in range(s):
        y[i] = neuronal_red_main(x[i], 500, 700, 0.02)

    plt.plot(x, y, color = "green")
    plt.scatter(x, y, color = "green", linewidths= 0.05)


    for i in range(s):
        y[i] = neuronal_red_main(x[i], 400, 700, 0.02)

    plt.plot(x, y, color = "orange")
    plt.scatter(x, y, color = "orange", linewidths= 0.05)


    for i in range(s):
        y[i] = neuronal_red_main(x[i], 300, 700, 0.02)

    plt.plot(x, y, color = "red")
    plt.scatter(x, y, color = "red", linewidths= 0.05)

    """

    t = np.arange(300, 500, 1)
    z = np.zeros(200)
    for j in range (350 , 500):
           for i in range(s):
                y[i] = neuronal_red_main(x[i], j)
                media = y.mean()
                z[j-300] = media

    """
    
    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)

    plt.title(name)
    plt.show()

def triDi():

    h = np.arange(0.2, 5.2, 0.2)
    i = np.arange(70,770,28)

    X,Y = np.meshgrid(h, i)

    Z = neuronalRedPreparation(X, Y)


    fig = plt.figure()
    ax = Axes3D(fig)

    surf = ax.plot_surface(X, Y, Z, cmap= cm.coolwarm, linewidths= 0, antialiaseds = False)
    fig.colorbar(surf, shrink = 0.5, aspect = 5)
    plt.xlabel("TasaDePrendizaje")
    plt.ylabel("Iteraciones")
    plt.show()

def neuronalRedPreparation(X, Y):
    z = X
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            z[i, j] = neuronal_red_main(8,  450, Y[i,j], X[i,j])
    return z


#regularized_logistic_regresion()
#logistic_regresion()
#part2_main()
#part3_main()
#neuronal_red_main()

#pintaFuncion(4, 25, 1, neuronal_red_main, "CapasOcultas", "PorcentajeAcierto", "RedNeuronal")
#neuronal_red_main(8,  450, 700, 10)
#triDi()

#pintaFunciontasaDeAprendizajeDependiente()
#pintaFuncionCapasOcultasDependiente()
#pintaFuncionIteracionesDependiente()
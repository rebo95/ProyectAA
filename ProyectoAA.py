import numpy as np
import matplotlib.pyplot as plt
import math
from math import e
from pandas.io.parsers import read_csv
from sklearn.preprocessing import PolynomialFeatures

import scipy.optimize as opt

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


#_______________________________________________________________________________________
#Regresión lineal
#_______________________________________________________________________________________

def normaliza(X):

    X_normalizada = np.zeros((X.shape[0], X.shape[1]))

    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])
    
    np.mean(X, axis = 0, out = mu)
    np.std(X, axis = 0, out = sigma)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X_normalizada[i,j] = (X[i,j] - mu[j])/sigma[j] 
    
    return X_normalizada, mu, sigma

def H_Theta(X, Z): #Hipótesis del mocelo lineal vectorizada 
    return np.dot(X, Z)

def funcion_coste(X, Y, Theta): #funcion de costes vectorizada
    H = H_Theta(X,Theta)
    Aux = (H-Y)**2
    sumatory = Aux.sum()/(2 * len(X))
    return sumatory


def normalEcuation(X,Y): #nos da los valores de theta que mejor se ajustan a nuestra regresión lineal

    #theta = (XT * X)**-1 * XT * Y 

    XT = np.transpose(X)

    XT__X = np.matmul(XT, X)

    XT__X_Inv = np.linalg.pinv(XT__X, rcond = 10**(-15))

    XT__X_Inv__XT = np.matmul(XT__X_Inv, XT)

    thetas = np.matmul(XT__X_Inv__XT, Y)

    return thetas

def descenso_gradiente(X, Y, alpha):
    
    m = X.shape[0]

    #construimos matriz Z
    th0 = 0.0
    th1 = 0.0
    th_n = 0.0

    Z = np.zeros(X.shape[1]) #estamos tomando la dimensión de la X con la columna de 1, de tal manera que esta si coincide con el valor de ladimension que tiene que tener el vector de thetas

    Z_ = np.zeros(X.shape[1] - 1)

    alpha_m = (alpha/m)

    Thetas = np.array([Z]) #almacena los thetas que forman parte de la hipotesis h_theta
    Costes = np.array([]) #almacena los costes obtenidos durante el descenso de gradiente
 
    for i in range(1500):

        #Calculo de Theta 0
        #Sumatorio para el calculo de Theta0
        sum1 = H_Theta(X, Z) - Y
        sum1_ = sum1.sum()
        th0 -= alpha_m * sum1_

        #Calculo Theta 1, 2, 3 ... n
        #Sumatorio para el calculo de Thetan
        for k in range(X.shape[1] - 1):
            sum2 =  (H_Theta(X, Z) - Y) * X[:, k + 1] # sería interesante ver cual es el resultado de probar con el uso de dot en vez del producto valor a valor
            sum2_ = sum2.sum()
            th_n -= alpha_m * sum2_ #vamos calculando cada uno de los thn
            Z_[k] = th_n #almacenamos los thn calculados en un vector provisional


        #Actualizamos los nuevos thetas del vector Z    
        Z[0] = th0
    
        for p in range(X.shape[1]-1):
            Z[p+1] = Z_[p]

        Thetas = np.append(Thetas, [Z], axis= 0)

        #funcion de costes
        J = funcion_coste(X,Y, Z)

        Costes = np.append(Costes, [J], axis = 0)

    return Thetas, Costes


def resuelve_problema_regresion_varias_variables():
    
    cancer_data = data_csv("data.csv")
    X, Y = data_builder(cancer_data)
    
    
    X_normalizada, mu, sigma = normaliza(X)


    X = np.hstack([np.ones([np.shape(X)[0], 1]), X])
    X_shape_1 = np.shape(X_normalizada)[0]


    X_normalizada = np.hstack([np.ones([X_shape_1, 1]), X_normalizada]) #le añadimos la columna de unos a la matriz ya normalizada

    Thetas, Costes = descenso_gradiente(X_normalizada, Y, 0.0022) #los valores de theta aquí son los obtenidos normalizando la matriz, esto es, necesitamos "desnormalizarlos"
    Thetas_normal_Ecuation = normalEcuation(X, Y)

    shape_thetas = np.shape(Thetas)[0]-1
    first_cancer_values = X[5, :]

    first_cancer_not_normalizes_values = convert_to_list(first_cancer_values)
    prediccion_normal_ecuation = H_Theta(first_cancer_not_normalizes_values ,Thetas_normal_Ecuation)


    first_cancer_normalizes_values = normalized_test_value(mu, sigma, first_cancer_not_normalizes_values)
    prediccion_gradiente_descendiente = H_Theta(first_cancer_normalizes_values, Thetas[shape_thetas])

    pn, pg = prediction_vectors(X, Thetas_normal_Ecuation, Thetas, mu, sigma)
    print(pn)
    print(pg)
   # prediccion_normal_ecuation = H_Theta( second_cancer_not_normalized_values ,Thetas_normal_Ecuation)

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


    
    
def normalized_test_value(mu, sigma, row_list) :

    dim = mu.shape[0]
    normalized_values = []
    normalized_values.append(1)

    for i in range(1, dim +1):
        normalized_values.append((row_list[0][i] - mu[i-1])/sigma[i-1])

    normalized_values = [normalized_values]
    return normalized_values


def convert_to_list(row):

    values_list = []
    for i in range(row.shape[0]):
        values_list.append(row[i])

    values_list = [values_list]

    return values_list




#_______________________________________________________________________________________
#Regresión logística
#_______________________________________________________________________________________
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


    return J


#_____________________________
#Función de coste regularizado
#_____________________________

def cost_regularized(Thetas, X, Y, h):

    m = X.shape[0] 
    Thetas_ = Thetas

    #J(θ) = (cost(Thetas, X, Y)) + D
    #J(θ) = [−(1/m) * ((log (g(Xθ)))T * y + (log (1 − g(Xθ)))T * (1 − y)))] + (λ/2m)*E(Theta^2)

    cost_ = cost(Thetas, X, Y)
   
    #D
    Thetas_ = np.power(Thetas_, 2)

    D = h/(2*m) * np.sum(Thetas_)

    J_regularized = (cost_) + D

    return J_regularized

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
    

#____________________________________
#Función thetas optimos regularizados
#____________________________________
def optimized_parameters_regularized(Thetas, X, Y, h):

    result = opt.fmin_tnc(func = cost_regularized, x0 = Thetas, fprime = gradient_regularized, args = (X, Y, h) )
    theta_opt = result[0]

    return theta_opt


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

    cost_ = cost(Thetas, X_V, Y_)
    gradient_ = gradient(Thetas, X_V, Y_)

    #testing and comparing if the resoults are correct having in mind the resoults given in the assignment document

    print(cost_)
    print(gradient_)

    optimized_thetas = optimized_parameters(Thetas, X_V, Y_)#Optimus cost is obtained by calling the cost ecuation using the optimized thetas

    print("Thetas optimos: ", optimized_thetas)
    cost_ = cost(optimized_thetas, X_V, Y_)
    print("Coste optimo : ", cost_)

#________________________________________
#Función Regresión logística regularizada
#________________________________________
def regularized_logistic_regresion(h = 1):

    cancer_data = data_csv("data.csv")
    X_, Y_ = data_builder(cancer_data)
    
    poly = PolynomialFeatures(degree=6)
    X_poly = poly.fit_transform(X_)
    
    #theta and alpha inicialization
    Thetas = np.zeros(X_poly.shape[1])

    cost_r = cost_regularized(Thetas, X_poly, Y_, h)
    gradient_r = gradient_regularized(Thetas, X_poly, Y_, h)
    print(cost_r)
    print(gradient_r)

    optimized_thetas_regularized = optimized_parameters_regularized(Thetas, X_poly, Y_, h)




#_______________________________________________________________________________________
#Regresión logística
#_______________________________________________________________________________________


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

resuelve_problema_regresion_varias_variables()
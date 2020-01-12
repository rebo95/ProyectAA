import numpy as np
import matplotlib.pyplot as plt
import math
from math import e
from pandas.io.parsers import read_csv

def data_csv(file_name):
    "Takes the data from the csv file and tranfers it to a numpy array"

    values_ = read_csv(file_name, header=None).values

    return values_.astype(float)


def data_builder(data):

    X = data[:, 2:]
    Y = data[:, 1]

    return X, Y



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
    print(y)

main()
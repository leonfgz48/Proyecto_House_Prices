from Modulo_scrip import *
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

def lee_train_data():
    train_data = pd.read_csv("../house-prices-data/train.csv") #Ruta de Datos
    
###Función_DATA
def Fill_na(head):
    train_data = pd.read_csv("../house-prices-data/train.csv") #Ruta de Datos

    train_data[head].fillna("No", inplace=True)

#Función_data (listo)
def fill_all_missing_values(data):
    for col in data.columns:
        if ((data[col].dtype == 'float64') or (data[col].dtype == 'int64')):
            data[col].fillna(data[col].mean(), inplace=True)
        else:
            data[col].fillna(data[col].mode()[0], inplace=True)

#Función_Processing
def encode_catagorical_columns(train, test):
    for col in Level_col:
        train[col] = encoder.fit_transform(train[col])
        test[col] = encoder.transform(test[col])
###

Level_col = ['Street', 'BldgType', 'SaleType', 'CentralAir']
encoder = LabelEncoder()

def encode_catagorical_columns(train, test):
    for col in Level_col:
        train[col] = encoder.fit_transform(train[col])
        test[col] = encoder.transform(test[col])
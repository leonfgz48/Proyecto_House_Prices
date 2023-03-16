## Inicio de Modelo 
#Libreria
import logging
#Configure logging
logging.basicConfig(
    filename='../logs/Model_Tarea_03.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

from Modulo_scrip import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
####

logging.info('*****Beggining Process*****')
# train_data = lee_train_data()


try:
    train_data = pd.read_csv("../house-prices-data/clean_train.csv") #Ruta de Datos
except Exception as e:
  logging.exception("Error al guardar submission.csv")
# train_data = pd.read_csv("../house-prices-data/clean_train.csv") #Ruta de Datos

try:
    test_data = pd.read_csv("../house-prices-data/test.csv") #Ruta de Datos
except Exception as e:
  logging.exception("Error al guardar submission.csv")
# test_data = pd.read_csv("../house-prices-data/test.csv") #Ruta de Datos

test_ids = test_data['Id']

test_data = pd.read_csv("../house-prices-data/clean_test.csv") #Ruta de Datos
print("Shape:", train_data.shape)

y = train_data['SalePrice']
X = train_data.drop(['SalePrice'], axis=1)

candidate_max_leaf_nodes = [250]
# model = LinearRegression()

for node in candidate_max_leaf_nodes:
    model = RandomForestRegressor(max_leaf_nodes=node,)
    model.fit(X, y)
    score = cross_val_score(model, X, y, cv=10)
    print(score.mean())

price = model.predict(test_data)
submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": price
})


try:
    submission.to_csv("submission.csv", index=False)
except Exception as e:
  logging.exception("Error al guardar submission.csv")

# submission.to_csv("submission.csv", index=False)
submission.sample(10)
logging.info('*****End Process*****')

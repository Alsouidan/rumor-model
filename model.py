from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import pandas as pd
def preprocess_feature(feature,data):
    ohe =OneHotEncoder()
    categories = np.array(
        list(set(data[feature].astype(str).values))).reshape(-1, 1)
    print(categories)
    print(data[feature].astype(str).values.reshape(-1,1))
    ohe.fit(data[feature].astype(str).values.reshape(-1, 1))
    OneHotEncoder(categories=categories,
                  dtype= np.float64 , handle_unknown='error', sparse=True)
    print(ohe.transform(data[feature].values.reshape(-1,1)))
    return ohe.transform(data[feature].values.reshape(-1,1))
def one_hot_pandas(features,df):
    for feature in features:
        df = pd.concat([df, pd.get_dummies(
            df[feature], prefix=feature)], axis=1)
    return df
data=pd.read_csv('train_data2.csv')
target_column=['result','gender','age','verified']
data=one_hot_pandas(['gender', 'age', 'verified'],data)
print(data)
predictors = list(set(list(data.columns))-set(target_column))
data[predictors]=(data[predictors]-data[predictors].mean())/data[predictors].max()
# data = data.sample(frac=1)
# preprocess_feature
y=data['result']
X = data.drop(target_column,axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y)
mlp = MLPClassifier(hidden_layer_sizes=(20, 20, 20,20,20,20,20), max_iter=10000,verbose=True)
mlp.fit(X, y)
predictions = mlp.predict(X_test)
print(classification_report(y_test,predictions))



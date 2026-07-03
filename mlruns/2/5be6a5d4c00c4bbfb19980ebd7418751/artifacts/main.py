import mlflow

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Wine_Quality_Classification")

wine=load_wine()

x=wine.data
y=wine.target 

# print(x)
# print(y)

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

max_depth=7
n_estimators=30


with mlflow.start_run():

    rf=RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators,random_state=42)
    rf.fit(X_train,y_train)

    y_pred=rf.predict(X_test)
    acc=accuracy_score(y_test,y_pred)

    mlflow.log_metric('accuracy',acc)
    mlflow.log_metric('max_depth',max_depth)
    mlflow.log_metric('n_estimators',n_estimators)


    # creating confusion matrix 

    cm=confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')


    # save plot 
    plt.savefig("confusion-matrix.png")

    # log artifacts using mlflow  
    mlflow.log_artifact("confusion-matrix.png")
    mlflow.log_artifact(__file__)

    # tages 
    mlflow.set_tags({"Author" : 'vikash', "Project":"Wine Classification"})

    # log the model 
    mlflow.sklearn.log_model(rf,"Random forest model")
    






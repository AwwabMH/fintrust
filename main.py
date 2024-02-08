from flask import Flask, render_template, request, url_for
import pickle
from xgboost import XGBClassifier
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
# from catboost import Pool, CatBoostRegressor, CatBoostClassifier, cv
# from catboost.utils import get_roc_curve


app = Flask(__name__)

model_names = ["ADA", "DCT", "GBC", "GNB", "KNN", "LDA", "LR", "MLP", "QDA", "RFC", "SVC", "XGB"]

model_weights = [0.03872549019607843, 0.1823529411764706, 0.027941176470588237, 0.03725490196078432, 0.09754901960784315, 0.06225490196078432, 0.04901960784313726, 0.09215686274509806, 0.07450980392156864, 0.17794117647058824, 0.01911764705882353, 0.1411764705882353]

models = []

with open('modelXGB.pkl', 'rb') as file: modelXGB = pickle.load(file)
with open('modelADA.pkl', 'rb') as file: modelADA = pickle.load(file)
# with open('modelCB.pkl', 'rb') as file: modelCB = pickle.load(file)
with open('modelDCT.pkl', 'rb') as file: modelDCT = pickle.load(file)
with open('modelGBC.pkl', 'rb') as file: modelGBC = pickle.load(file)
with open('modelGNB.pkl', 'rb') as file: modelGNB = pickle.load(file)
with open('modelKNN.pkl', 'rb') as file: modelKNN = pickle.load(file)
with open('modelLDA.pkl', 'rb') as file: modelLDA = pickle.load(file)
with open('modelLR.pkl', 'rb') as file: modelLR = pickle.load(file)
with open('modelMLP.pkl', 'rb') as file: modelMLP = pickle.load(file)
with open('modelQDA.pkl', 'rb') as file: modelQDA = pickle.load(file)
with open('modelRFC.pkl', 'rb') as file: modelRFC = pickle.load(file)
with open('modelSVC.pkl', 'rb') as file: modelSVC = pickle.load(file)

models.append(modelADA)
# models.append(modelCB)
models.append(modelDCT)
models.append(modelGBC)
models.append(modelGNB)
models.append(modelKNN)
models.append(modelLDA)
models.append(modelLR)
models.append(modelMLP)
models.append(modelQDA)
models.append(modelRFC)
models.append(modelSVC)
models.append(modelXGB)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction', methods=['POST', 'GET'])
def predict():
    
    predictions = []
    
    fintrust_prediction = 0
    
    amount = float(request.form['amount'])
    oldbalanceOrg = float(request.form['oldbalanceOrg'])
    newbalanceOrig = float(request.form['newbalanceOrig'])
    oldbalanceDest = float(request.form['oldbalanceDest'])
    newbalanceDest = float(request.form['newbalanceDest'])
    
    
    for model, weight in zip(models, model_weights):
        if model != modelSVC:
            number = round(model.predict_proba([[amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]])[0][1]*100, 2)
            fintrust_prediction += (number * weight)
            predictions.append(str(number) + "% Fraud")
        else:
            number = model.predict([[amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]])[0]*100
            fintrust_prediction += (number * weight)
            predictions.append(str(number) + "% Fraud")

    fintrust_prediction = str(round(fintrust_prediction, 2)) + "% Fraud"
            

    return render_template('result.html', 
                           prediction0=fintrust_prediction,
                           prediction1=predictions[0], 
                           prediction2 = predictions[1], 
                           prediction3 = predictions[2], 
                           prediction4 = predictions[3], 
                           prediction5 = predictions[4], 
                           prediction6 = predictions[5], 
                           prediction7 = predictions[6], 
                           prediction8 = predictions[7], 
                           prediction9 = predictions[8],
                           prediction10 = predictions[9],
                           prediction11 = predictions[10],
                           prediction12 = predictions[11],
                           
    )

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        
        file = open('output.txt', 'a')

        file.write("\n" + name + "\n" + email + "\n" + message)

        file.close()
        
        return render_template('success.html')  
    else:
        return render_template('contact.html')  


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run()
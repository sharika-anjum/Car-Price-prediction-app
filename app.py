from flask import Flask, request, render_template, url_for
import requests
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.metrics import r2_score

import pickle

# Reference dictionaries
fuel = pickle.load(open('fuel.pkl', 'rb'))
fl = list(fuel.keys())
company = pickle.load(open('company.pkl', 'rb'))
comp = list(company.keys())
comp_new = comp
year = pickle.load(open('year.pkl', 'rb'))
yr = list(set(year))
# Regressor
reg = pickle.load(open('linreg1.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html', comp=comp_new, yr=yr)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        brand = request.form.get('company')
        yop = int(request.form.get('year'))
        dist = int(request.form.get('kilo_driven'))
        ful = request.form.get('fuel_type')
        # Encoding brand and fuel
        brand_enc = company[brand]
        fuel_enc = fuel[ful]
        inp = [brand_enc, yop, dist,fuel_enc]
        # Prediction results
        price = (int(reg.predict(np.array(inp).reshape(1, 4))[0]))*10
        return render_template('index.html', price=np.round(price), check=1)
    else:
        return render_template('index.html', check=0)


if __name__=="__main__":
    app.run(debug=True)

from flask import Flask, render_template, request
from sklearn import preprocessing

from tensorflow.keras.models import load_model
model_wine = load_model("modelWine.h5")

import pandas as pd
import numpy as np

admissions = pd.read_csv('minmax.csv', delimiter=";", header=None)
datos = admissions.values
datos = np.array(datos, "float32")

def pred(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, datos):
    wine_array = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13]
    wine_array = np.array(wine_array, "float32")
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    datos = np.concatenate((datos, [wine_array]), axis=0)
    datos = min_max_scaler.fit_transform(datos)
    datos = np.array([datos[-1]])
    prediction = model_wine.predict([datos]).argmax(axis=1)
    return prediction

app = Flask(__name__)

@app.route("/")
def hello():
    return render_template("index.html")
    
@app.route("/sub", methods = ["POST"])
def submit():
    # HTML -- py
    if request.method == "POST":
        p1 = float(request.form["p1"])
        p2 = float(request.form["p2"])
        p3 = float(request.form["p3"])
        p4 = float(request.form["p4"])
        p5 = float(request.form["p5"])
        p6 = float(request.form["p6"])
        p7 = float(request.form["p7"])
        p8 = float(request.form["p8"])
        p9 = float(request.form["p9"])
        p10 = float(request.form["p10"])
        p11 = float(request.form["p11"])
        p12 = float(request.form["p12"])
        p13 = float(request.form["p13"])

    prediction = int(pred(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, datos))+1

    # py -- HTML
    
    return render_template("sub.html", result = prediction)

if __name__ == "__main__":
    app.run()
import pickle
from flask import Flask,request,app,jsonify,url_for,render_template

import numpy as np
import pandas as pd

app = Flask(__name__) # starting point of the application name of the app
regmodel = pickle.load(open('regmodel.pkl','rb')) # load the model
scalar = pickle.load(open('scalar.pkl','rb')) # loading the scalar tranform

@app.route('/')   # redirecting to the home page
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])

def predict_api():
    data = request.json['data']  # when i hit predict_api   (input)
    print(data)

    # standardizing the data

    print(np.array(list(data.values())).reshape(1,-1)) # converting it into list because its in json format
    new_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))

    output = regmodel.predict(new_data)

    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():

    data = [float(x) for x in request.form.values()]
    final_inp = scalar.transform(np.array(data).reshape(1,-1))
    print(final_inp)

    output = regmodel.predict(final_inp)[0]

    return render_template("home.html",prediction_text="The House Price Predicted is {}".format(output))




if __name__=="__main__":
    app.run(debug=True)

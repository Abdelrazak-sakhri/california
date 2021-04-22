import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import joblib
import pandas as pd
#data
from sklearn import datasets




app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html',
    )


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', 
    s1='Revenus moyens des habitants du quatier : {}'.format(int_features[0]),
    s2='Age moyen des maisons dans le quartier : {}'.format(int_features[1]),
    s3='Nombre moyen de pièces : {}'.format(int_features[2]),
    s4='Nombre moyen de chambres : {}'.format(int_features[3]), 
    s5='Population du quartier : {}'.format(int_features[4]),
    s6='Nombre moyen d\'occupant : {}'.format(int_features[5]),
    s7='Latitude : {}'.format(int_features[6]),
    s8='Longitude : {}'.format(int_features[7]),
    
    prediction_text='Le bien est estimé à {}'.format(output))


@app.route('/predict_api', methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)

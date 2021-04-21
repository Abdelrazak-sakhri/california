import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


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
    s1='MedInc : {}'.format(int_features[0]),
    s2='HouseAge : {}'.format(int_features[1]),
    s3='AveRooms : {}'.format(int_features[2]),
    s4='AveBedrms : {}'.format(int_features[3]), 
    s5='Population : {}'.format(int_features[4]),
    s6='AveOccup : {}'.format(int_features[5]),
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

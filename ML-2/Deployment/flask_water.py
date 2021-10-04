import os
mycwd = 'C:\\Users\\jana\\Documents\\Machine Learning 2\\Project\\Deployment'
os.chdir(mycwd)
os.getcwd()

# an object of WSGI application
#from flask import Flask	
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
app = Flask(__name__) # Flask constructor

variety_mappings = {0: 'Not-Drinkable', 1: 'Drinkable'}



#app = Flask(__name__)
xgb = pickle.load(open('logreg1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/predict',methods=['GET'])
def predform():
    return render_template('predict.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    input_features = [float(x) for x in request.form.values()]
    final_features = np.array(input_features)
    query = final_features.reshape(1,-1)
    output = variety_mappings[xgb.predict(query)[0]]
    return render_template('predict.html', prediction_text='Water is: {}'.format(output))


if __name__ == "__main__":
    app.run(debug=False)

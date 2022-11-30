from flask import Flask,request, url_for, redirect, render_template, jsonify
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

service_fees = pd.read_csv('datasets/Service_fees.csv')
service_fees = service_fees.sort_values(by='encoding')

model = pickle.load(open('model/model.pkl', 'rb'))
cols = ['country','amount','PSP', '3D_secured', 'card', 'weekday', 'hour']

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    input_data = pd.DataFrame([final], columns = cols)    
    return render_template('home.html',pred='Best possible PSP is: {}'.format(model_prdedict(input_data)))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    input_data = pd.DataFrame([data])
    output = model_prdedict(input_data)
    return jsonify(output)


def model_prdedict(input_df): 
    scores = []
    for x in range(0,4):
        input_df['PSP'] = x
        proba = model.predict_proba(input_df)[0][1]
        score = get_score(round(proba*100,0)) * float(service_fees.query('encoding=='+str(x))['score'].item())
        scores.append({'provider':service_fees.query('encoding=='+str(x))['name'].item(),'proba':round(proba,2), 'score':score}) 
    scores = pd.DataFrame(scores).sort_values(by='score',ascending=False)
    return {'Provider':scores[0:1]['provider'].item(), 'Score': scores[0:1]['score'].item()}
    

    
def get_score(value):
    if value in range(0,10):
        return 1
    elif value in range(11,20):
        return 2
    elif value in range(21,30):
        return 3
    elif value in range(31,40):
        return 4
    elif value in range(41,50):
        return 5
    elif value in range(51,60):
        return 6
    elif value in range(61,70):
        return 7
    elif value in range(71,80):
        return 8
    elif value in range(81,90):
        return 9
    elif value in range(91,100):
        return 10    

if __name__ == '__main__':
    app.run(debug=True)

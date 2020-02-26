import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = np.round(prediction[0], 2)
    # find biggest number
    #max1=0
    for i in range(len(output)):
      if output[i] >= max(output):
        output[i] = 1
      else:
        output[i] = 0
        
    if output[0] == 1:
      grade = 'Excellent'
    elif output[1] == 1:
      grade = 'Fail'
    elif output[2] == 1:
      grade = 'Good'
    elif output[3] == 1:
      grade = 'Poor'
    else:
      grade = 'Very Good'
      

    #return render_template('index.html', prediction_text='Predicted Grade is {}'.format(grade))
    grade_json = jsonify({"grade" : grade})
    return grade_json


if __name__ == "__main__":
    app.run(debug=True)

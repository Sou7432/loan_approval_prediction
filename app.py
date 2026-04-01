from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
app = Flask(__name__)
with open('loan.pkl', 'rb') as model_file:
    ml_model = pickle.load(model_file)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/process', methods=['POST'])
def process():
    if request.method == 'POST':
        try:
            features =[
                float(request.form['age']),
                float(request.form['income']),
                 float(request.form['coincome']),
                float(request.form['loanamount']),
                float(request.form['loanduration']),
                float(request.form['creditscore'])
                 ]
            input_data = np.array(features).reshape(1,-1)
            result = ml_model.predict(input_data)[0]
            loan_approval = 'Approved' if result == 1 else 'Not Approved'
            return render_template('result.html', prediction=loan_approval)
        except Exception as e:
            return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)

    
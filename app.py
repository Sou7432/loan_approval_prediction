from flask import Flask, request, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model safely (important for Render deployment)
model_path = os.path.join(os.path.dirname(__file__), "loan.pkl")
with open(model_path, "rb") as model_file:
    ml_model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        features = [
            float(request.form.get('age', 0)),
            float(request.form.get('income', 0)),
            float(request.form.get('coincome', 0)),
            float(request.form.get('loanamount', 0)),
            float(request.form.get('loanduration', 0)),
            float(request.form.get('creditscore', 0))
        ]

        input_data = np.array(features).reshape(1, -1)
        result = ml_model.predict(input_data)[0]

        loan_approval = 'Approved ✅' if result == 1 else 'Not Approved ❌'

        return render_template('result.html', prediction=loan_approval)

    except Exception as e:
        return f"Error: {str(e)}"

# IMPORTANT: Render requires this
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render assigns PORT
    app.run(host="0.0.0.0", port=port)

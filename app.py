import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from flask import Flask, request, render_template_string

# Sample dataset (can be expanded or replaced with real-world data)
data = {
    'Salary': [3000, 5000, 2500, 7000, 4000, 8000, 2200, 6000],
    'LoanAmount': [100, 200, 120, 300, 150, 250, 100, 180],
    'Credit_History': [1, 1, 0, 1, 1, 1, 0, 1],
    'Loan_Status': ['Y', 'Y', 'N', 'Y', 'Y', 'Y', 'N', 'Y']
}

# Create DataFrame
df = pd.DataFrame(data)

# Prepare features and target
X = df[['Salary', 'LoanAmount', 'Credit_History']]
y = df['Loan_Status']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Loan Approval Predictor</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            form { max-width: 400px; margin: auto; }
            input, button { display: block; width: 100%; margin: 10px 0; padding: 10px; }
            button { background-color: #4CAF50; color: white; border: none; cursor: pointer; }
            button:hover { background-color: #45a049; }
            .result { margin-top: 20px; padding: 10px; border: 1px solid #ddd; }
        </style>
    </head>
    <body>
        <h1>Loan Approval Predictor</h1>
        <form action="/predict" method="post">
            <label for="salary">Salary:</label>
            <input type="number" id="salary" name="salary" required>
            
            <label for="loan_amount">Loan Amount:</label>
            <input type="number" id="loan_amount" name="loan_amount" required>
            
            <label for="credit_history">Credit History (1 for Yes, 0 for No):</label>
            <input type="number" id="credit_history" name="credit_history" min="0" max="1" required>
            
            <button type="submit">Predict</button>
        </form>
    </body>
    </html>
    ''')

@app.route('/predict', methods=['POST'])
def predict():
    salary = int(request.form['salary'])
    loan_amount = int(request.form['loan_amount'])
    credit_history = int(request.form['credit_history'])
    
    # Make prediction
    prediction = model.predict([[salary, loan_amount, credit_history]])[0]
    result = "Approved" if prediction == 'Y' else "Not Approved"
    
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Prediction Result</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; text-align: center; }
            .result { margin-top: 20px; padding: 20px; border: 1px solid #ddd; display: inline-block; }
            a { display: inline-block; margin-top: 20px; padding: 10px 20px; background-color: #4CAF50; color: white; text-decoration: none; }
            a:hover { background-color: #45a049; }
        </style>
    </head>
    <body>
        <h1>Loan Approval Prediction</h1>
        <div class="result">
            <h2>Result: {{ result }}</h2>
        </div>
        <a href="/">Predict Again</a>
    </body>
    </html>
    ''', result=result)

if __name__ == '__main__':
    app.run(debug=True)

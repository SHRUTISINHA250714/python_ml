from flask import Flask, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import pandas as pd

app = Flask(__name__)
CORS(app)

# Load and process the dataset
df = pd.read_csv("hardware_inventory_realistic_prices.csv")

X = df[["Category", "Quantity", "Condition", "Maintenance_Charge", "Average_Maintenance_Cost", "Item_Age", "Return_Duration"]]
y = df["Price"]

# One-hot encoding for categorical variables
X = pd.get_dummies(X, columns=["Category", "Condition"], drop_first=True).astype(int)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
price_model = RandomForestRegressor(n_estimators=100, random_state=42)
price_model.fit(X_train, y_train)

# Make predictions and calculate Mean Squared Error
y_pred = price_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Convert y_test and y_pred to lists so they can be returned as JSON
y_test_list = y_test.head(30).tolist()
y_pred_list = y_pred[:30].tolist()
print(y_test_list)
print(y_pred_list)

# Define a route that returns the data
@app.route('/get_chart_data')
def get_chart_data():
    return jsonify({
        "y_test": y_test_list,
        "y_pred": y_pred_list
    })
 


if __name__ == '__main__':
    app.run()
# from flask import Flask, json
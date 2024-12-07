# # from flask import Flask
# # from flask_cors import CORS
# # from flask import request, jsonify
# # from sklearn.ensemble import RandomForestRegressor
# # from sklearn.metrics import mean_squared_error
# # from sklearn.model_selection import train_test_split
# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt

# # df = pd.read_csv("C:\\Users\\Lenovo\\Desktop\\SIH_SurakshaSanchay\\src\\app\\Ml\\hardware_inventory_realistic_prices.csv")

# # X = df[["Category", "Quantity", "Condition", "Maintenance_Charge", "Average_Maintenance_Cost", "Item_Age", "Return_Duration"]]
# # y = df["Price"]

# # X = pd.get_dummies(X, columns=["Category", "Condition"], drop_first=True).astype(int)
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # price_model = RandomForestRegressor(n_estimators=100, random_state=42)
# # price_model.fit(X_train, y_train)

# # y_pred = price_model.predict(X_test)
# # mse = mean_squared_error(y_test, y_pred)
# # print(mse)

# # app = Flask(__name__)
# # CORS(app)
# # from flask import Flask
# # from flask_cors import CORS  # Make sure to import CORS
# # from flask import request, jsonify
# # from sklearn.ensemble import RandomForestRegressor
# # from sklearn.metrics import mean_squared_error
# # from sklearn.model_selection import train_test_split
# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt

# # # Load and process the dataset
# # df = pd.read_csv("C:\\Users\\Lenovo\\Desktop\\SIH_SurakshaSanchay\\src\\app\\Ml\\hardware_inventory_realistic_prices.csv")

# # X = df[["Category", "Quantity", "Condition", "Maintenance_Charge", "Average_Maintenance_Cost", "Item_Age", "Return_Duration"]]
# # y = df["Price"]

# # # One-hot encoding for categorical variables
# # X = pd.get_dummies(X, columns=["Category", "Condition"], drop_first=True).astype(int)

# # # Splitting data into training and testing sets
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # # Train the model
# # price_model = RandomForestRegressor(n_estimators=100, random_state=42)
# # price_model.fit(X_train, y_train)

# # # Make predictions and calculate Mean Squared Error
# # y_pred = price_model.predict(X_test)
# # mse = mean_squared_error(y_test, y_pred)
# # print(mse)

# # plt.plot(y_test.head(30), y_pred[:30], marker='o', color="blue")
# # plt.title("Actual Prices vs Predicted Prices")
# # plt.xlabel("Actual Prices")
# # plt.ylabel("Predicted Prices")
# # plt.show()

# # # Initialize Flask app and enable CORS
# # app = Flask(__name__)
# # CORS(app)  # This enables CORS for the entire Flask app
# # from flask import Flask, jsonify
# # from flask_cors import CORS
# # from sklearn.ensemble import RandomForestRegressor
# # from sklearn.metrics import mean_squared_error
# # from sklearn.model_selection import train_test_split
# # import numpy as np
# # import pandas as pd

# # app = Flask(__name__)
# # CORS(app)

# # # Load and process the dataset
# # df = pd.read_csv("C:\\Users\\Lenovo\\Desktop\\SIH_SurakshaSanchay\\src\\app\\Ml\\hardware_inventory_realistic_prices.csv")

# # X = df[["Category", "Quantity", "Condition", "Maintenance_Charge", "Average_Maintenance_Cost", "Item_Age", "Return_Duration"]]
# # y = df["Price"]

# # # One-hot encoding for categorical variables
# # X = pd.get_dummies(X, columns=["Category", "Condition"], drop_first=True).astype(int)

# # # Splitting data into training and testing sets
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # # Train the model
# # price_model = RandomForestRegressor(n_estimators=100, random_state=42)
# # price_model.fit(X_train, y_train)

# # # Make predictions and calculate Mean Squared Error
# # y_pred = price_model.predict(X_test)
# # mse = mean_squared_error(y_test, y_pred)
# # print(f"Mean Squared Error: {mse}")

# # # Convert y_test and y_pred to lists so they can be returned as JSON
# # y_test_list = y_test.head(30).tolist()
# # y_pred_list = y_pred[:30].tolist()
# # print(y_test_list)
# # print(y_pred_list)

# # # Define a route that returns the data
# # @app.route('/get_chart_data')
# # def get_chart_data():
# #     return jsonify({
# #         "y_test": y_test_list,
# #         "y_pred": y_pred_list
# #     })

# # if __name__ == '__main__':
# #     app.run(debug=True)
# from flask import Flask, jsonify,render_template
# import pandas as pd
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)

# # Load datasets
# hardware_inventory_data = pd.read_csv('./hardware_inventory_realistic_prices.csv')
# budget_data = pd.read_csv('./budgetval.csv')

# @app.route('/try')
# def index():
#     return render_template('index.html')


# @app.route('/api/category_maintenance_cost', methods=['GET'])
# def category_maintenance_cost():
#     category_cost = hardware_inventory_data.groupby('Category')['Maintenance_Charge'].sum().reset_index()
#     return jsonify(category_cost.to_dict(orient='records'))


# @app.route('/api/condition_maintenance_cost', methods=['GET'])
# def condition_maintenance_cost():
#     condition_cost = hardware_inventory_data.groupby('Condition')['Maintenance_Charge'].sum().reset_index()
#     return jsonify(condition_cost.to_dict(orient='records'))


# @app.route('/api/age_group_maintenance_cost', methods=['GET'])
# def age_group_maintenance_cost():
#     bins = range(0, int(hardware_inventory_data['Item_Age'].max()) + 100, 100)
#     hardware_inventory_data['Age_Group'] = pd.cut(hardware_inventory_data['Item_Age'], bins)
#     grouped_data = hardware_inventory_data.groupby('Age_Group')['Average_Maintenance_Cost'].mean().reset_index()
#     grouped_data['Age_Group'] = grouped_data['Age_Group'].astype(str)
#     return jsonify(grouped_data.to_dict(orient='records'))


# @app.route('/api/budget_category_spending', methods=['GET'])
# def budget_category_spending():
#     budget_data['Total Cost'] = budget_data['Price'] + budget_data['Maintenance_Charge']
#     category_summary = budget_data.groupby('Category')[['Price', 'Maintenance_Charge', 'Total Cost']].sum().reset_index()
#     return jsonify(category_summary.to_dict(orient='records'))


# @app.route('/api/yearly_spending_trends', methods=['GET'])
# def yearly_spending_trends():
#     if 'Year' in budget_data.columns:
#         yearly_summary = budget_data.groupby('Year')[['Price', 'Maintenance_Charge', 'Total Cost']].sum().reset_index()
#         return jsonify(yearly_summary.to_dict(orient='records'))
#     return jsonify([])



from flask import Flask
app = Flask(__name__)

@app.route('/')
def index():
    return "Hello, Flask on Vercel!"

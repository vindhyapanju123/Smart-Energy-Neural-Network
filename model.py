import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

# Load dataset
data = pd.read_csv("energy_data.csv")

# Features (inputs) and target (output)
X = data[['temperature', 'irradiance', 'voltage', 'current']]
y = data['energy_output']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Normalize (scaling)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Neural Network Model
model = MLPRegressor(hidden_layer_sizes=(32,16), max_iter=500)
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Show clear results
print("\n===== SAMPLE PREDICTIONS =====")
for i in range(len(predictions)):
    original_input = scaler.inverse_transform([X_test[i]])
    print(f"\nInput (Temp, Irradiance, Voltage, Current): {original_input[0]}")
    print(f"Actual Energy: {y_test.iloc[i]}")
    print(f"Predicted Energy: {predictions[i]:.2f}")

# User input for live prediction
print("\n===== TEST WITH YOUR OWN INPUT =====")
temp = float(input("Enter Temperature: "))
irr = float(input("Enter Irradiance: "))
volt = float(input("Enter Voltage: "))
curr = float(input("Enter Current: "))

user_data = scaler.transform([[temp, irr, volt, curr]])
user_prediction = model.predict(user_data)

print(f"\nPredicted Energy Output: {user_prediction[0]:.2f}")

# Graph
plt.scatter(y_test, predictions)
plt.xlabel("Actual Energy")
plt.ylabel("Predicted Energy")
plt.title("Actual vs Predicted Energy")
plt.savefig("graph.png")
plt.show()

# Model score
print("\nModel Accuracy Score:", model.score(X_test, y_test))
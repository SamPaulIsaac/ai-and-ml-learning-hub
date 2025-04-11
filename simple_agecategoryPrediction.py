import numpy as np
from sklearn.linear_model import LogisticRegression

# Training Data (Features: Age, Labels: 0 = Child, 1 = Adult)
X = np.array([[5], [17], [18], [25], [30]])  # Features (ages)
y = np.array([0, 0, 1, 1, 1])  # Labels (Child = 0, Adult = 1)

# Train an ML Model
model = LogisticRegression()
model.fit(X, y)

# Predict on new ages
new_ages = np.array([[10], [20], [15], [40]])
predictions = model.predict(new_ages)

# Print results
for age, pred in zip(new_ages.flatten(), predictions):
    label = "Adult" if pred == 1 else "Child"
    print(f"Age {age}: {label}")



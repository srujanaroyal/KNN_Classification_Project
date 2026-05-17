import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os



# Load dataset
df = pd.read_csv("Iris.csv")

# Drop ID column
df.drop("Id", axis=1, inplace=True)

# Encode target
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])

# Features and target
X = df.drop("Species", axis=1)
y = df["Species"]

# Scale
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = KNeighborsClassifier(
    n_neighbors=5,
    metric='minkowski',
    p=2
)

model.fit(X_train, y_train)

# Accuracy
pred = model.predict(X_test)
score = accuracy_score(y_test, pred)

# ---------------- Streamlit UI ----------------

st.title("Iris Flower Prediction")
st.write("KNN Classification Model")

st.subheader(f"Model Accuracy: {score:.2f}")

sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.5)
sepal_width = st.slider("Sepal Width", 2.0, 5.0, 3.0)
petal_length = st.slider("Petal Length", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width", 0.1, 3.0, 1.2)

# Input
input_data = pd.DataFrame([[
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
]], columns=df.drop("Species", axis=1).columns)

# Scale input
scaled_input = scaler.transform(input_data)

# Predict
if st.button("Predict Species"):
    prediction = model.predict(scaled_input)

    species = le.inverse_transform(prediction)

    st.success(f"Predicted Species: {species[0]}")
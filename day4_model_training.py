import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
data = pd.read_csv("fraudTrain.csv")

# Drop irrelevant columns
data.drop(columns=['Unnamed: 0', 'cc_num', 'first', 'last', 'street',
                   'city', 'state', 'zip', 'trans_num'], inplace=True)

# Convert dates
data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
data['dob'] = pd.to_datetime(data['dob'])

# Feature engineering
data['trans_hour'] = data['trans_date_trans_time'].dt.hour
data['trans_day']  = data['trans_date_trans_time'].dt.dayofweek
data['age'] = data['trans_date_trans_time'].dt.year - data['dob'].dt.year

# Drop original date columns
data.drop(columns=['trans_date_trans_time', 'dob'], inplace=True)

# Encode categorical columns (separately)
for col in ['merchant', 'category', 'job']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))

# Scale amount
scaler = StandardScaler()
data['amt'] = scaler.fit_transform(data[['amt']])

# Split features and target
X = data.drop('is_fraud', axis=1)
y = data['is_fraud']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("✅ Model Trained Successfully")

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\n📊 Model Performance:")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))

print("\n🎉 Day 4 Completed Successfully!")
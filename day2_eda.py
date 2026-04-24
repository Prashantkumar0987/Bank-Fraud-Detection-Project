import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Save charts to file instead of opening a window
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("fraudTrain.csv")

# Count values
print("Transaction Counts:")
print(data['is_fraud'].value_counts())

# Bar chart - Fraud vs Normal Transactions
data['is_fraud'].value_counts().plot(kind='bar', color=['steelblue', 'tomato'])
plt.title("Fraud vs Normal Transactions")
plt.xlabel("Class (0 = Normal, 1 = Fraud)")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("fraud_distribution.png")
plt.close()
print("\nChart saved as 'fraud_distribution.png'")

# Fraud percentages
fraud = data['is_fraud'].value_counts()[1]
normal = data['is_fraud'].value_counts()[0]

print("\nFraud Percentage: ", round((fraud / len(data)) * 100, 2), "%")
print("Normal Percentage:", round((normal / len(data)) * 100, 2), "%")
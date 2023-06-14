import pandas as pd
import cupy as cp
import pickle

# Load the dataset
data = pd.read_csv('vgsales_categorized.csv')

from sklearn.preprocessing import LabelEncoder

# Convert non-numeric columns to numeric using label encoding
label_encoder = LabelEncoder()
data['Platform'] = label_encoder.fit_transform(data['Platform'])
data['Genre'] = label_encoder.fit_transform(data['Genre'])
data['Publisher'] = label_encoder.fit_transform(data['Publisher'])
data['Sales_Category'] = label_encoder.fit_transform(data['Sales_Category'])
# Split the dataset into features (X) and target (y)
X = cp.asarray(data[['Platform', 'Genre', 'Publisher']].values)  # Add more features as needed
y = cp.asarray(data['Sales_Category'].values)

y = y.astype(int)

# Calculate class probabilities
class_probs = cp.bincount(y) / len(y)

# Calculate feature probabilities for each class
feature_probs = cp.zeros((class_probs.size, X.shape[1]))
for cls in cp.unique(y):
    cls_indices = cp.where(y == cls)[0]
    cls_data = X[cls_indices]
    feature_probs[cls] = cls_data.sum(axis=0) / cls_data.size

# Save the trained model to a file
model = {'class_probs': class_probs.get(), 'feature_probs': feature_probs.get()}
with open('naive_bayes_model.pkl', 'wb') as file:
    pickle.dump(model, file)

import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('vgsales_categorized.csv')
data = data.astype(str)
# Extract the features (X) and the target variable (y)
X = data[['Platform', 'Genre', 'Publisher']]  # Add more features as needed
y = data['Sales_Category']

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Convert the string features to numerical representation using CountVectorizer
vectorizer = CountVectorizer()
X_encoded = vectorizer.fit_transform(X.apply(lambda x: ' '.join(x), axis=1))

# Train the Naive Bayes classifier
naive_bayes = MultinomialNB()
naive_bayes.fit(X_encoded, y_encoded)

# Save the trained model and label encoder to files
with open('naive_bayes_model.pkl', 'wb') as file:
    pickle.dump(naive_bayes, file)

with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)

# ... Later, in another session or script ...

# Load the saved model and label encoder from files
with open('naive_bayes_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

# Prepare new data for prediction
new_data = pd.DataFrame({'Platform': ['Wii'], 'Genre': ['Action'], 'Publisher': ['Nintendo']})  # Add more features as needed

# Convert the string features to numerical representation
new_data_encoded = vectorizer.transform(new_data.apply(lambda x: ' '.join(x), axis=1))

# Use the loaded model for prediction
predictions_encoded = loaded_model.predict(new_data_encoded)

# Convert the encoded predictions back to the original labels
predictions = label_encoder.inverse_transform(predictions_encoded)
print(predictions)

train_predictions = naive_bayes.predict(X_encoded)

# Decode the encoded predictions and ground truth labels
train_predictions_decoded = label_encoder.inverse_transform(train_predictions)
y_decoded = label_encoder.inverse_transform(y_encoded)

# Calculate the accuracy
accuracy = accuracy_score(y_decoded, train_predictions_decoded)
print("Training Accuracy:", accuracy)

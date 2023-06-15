import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

data = pd.read_csv('vgsales.csv')
data['Sales_Category'] = pd.qcut(data['Global_Sales'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
sales_bins = [0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
sales_labels = ["No Sales", "Minimal Sales", "Low Sales", "Moderate Sales", "High Sales", "Very High Sales", "Exceptional Sales", "Record-breaking Sales"]

data['Sales_Category'] = pd.cut(data['Global_Sales'], bins=sales_bins, labels=sales_labels)

data = data.astype(str)

X = data[['Platform', 'Genre', 'Publisher']]
y = data['Sales_Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_encoded = vectorizer.fit_transform(X_train.apply(lambda x: ' '.join(x), axis=1))
X_test_encoded = vectorizer.transform(X_test.apply(lambda x: ' '.join(x), axis=1))

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_encoded, y_train_encoded)

import pickle

with open('naive_bayes_model.pkl', 'wb') as file:
    pickle.dump(naive_bayes, file)

with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)

with open('naive_bayes_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

platform = ['Wii', 'PS2']
genre = ['Action', 'Role-Playing']
publisher = ['Nintendo', 'Activision']

new_data = pd.DataFrame({'Platform': platform, 'Genre': genre, 'Publisher': publisher})

new_data_encoded = vectorizer.transform(new_data.apply(lambda x: ' '.join(x), axis=1))

predictions_encoded = loaded_model.predict(new_data_encoded)
predictions = label_encoder.inverse_transform(predictions_encoded)

print([p for p in predictions])

train_predictions = naive_bayes.predict(X_train_encoded)
train_predictions_decoded = label_encoder.inverse_transform(train_predictions)
y_train_decoded = label_encoder.inverse_transform(y_train_encoded)

accuracy = accuracy_score(y_train_decoded, train_predictions_decoded)
print("Training Accuracy:", accuracy)

f1 = f1_score(y_train_decoded, train_predictions_decoded, average="weighted")
print("F1 Score:", f1)

recall = recall_score(y_train_decoded, train_predictions_decoded, average="weighted")
print("Recall:", recall)

precision = precision_score(y_train_decoded, train_predictions_decoded, average="weighted")
print("Precision:", precision)

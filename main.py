import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("sad_texts.csv")

# Pre-process text data
texts = df['text'].str.lower()
vectorizer = CountVectorizer()
text_vectors = vectorizer.fit_transform(texts)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(text_vectors, df['label'], test_size=0.2)

# Train the model
model = MultinomialNB()
model.fit(x_train, y_train)
#
# # Make predictions on test data
# y_pred = model.predict(x_test)

input_text = input("Enter a text: ")
input_features = vectorizer.transform([input_text])
prediction = model.predict(input_features)
label = prediction[0]
confidence = model.predict_proba(input_features)

print("The text is classified as:", "sad" if label == '1' else "not sad")
print("Confidence", confidence[0][1])

# Evaluate model performance
# acc = accuracy_score(y_test, y_pred)
# print("Accuracy:", acc)

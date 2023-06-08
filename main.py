import csv
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pandas as pd

import matplotlib.pyplot as plt

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load data
df = pd.read_csv("sad_texts.csv", header=None, names=['text', 'label'])

# Convert label column to numeric values
df['label'] = pd.to_numeric(df['label'], errors='coerce')

# Remove rows with missing label values
df = df.dropna(subset=['label'])

# Pre-process text data
texts = df['text'].str.lower()

# Remove special characters and numbers
texts = texts.apply(lambda x: re.sub(r'[^a-zA-Z]', ' ', x))

# Tokenize the text
texts = texts.apply(lambda x: x.split())

# Remove stopwords
stop_words = set(stopwords.words('english'))
texts = texts.apply(lambda x: [word for word in x if word not in stop_words])

# Lemmatize the words
lemmatizer = WordNetLemmatizer()
texts = texts.apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

# Join the words back into sentences
texts = texts.apply(lambda x: ' '.join(x))

# Vectorize the text data
vectorizer = CountVectorizer()
text_vectors = vectorizer.fit_transform(texts)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(text_vectors, df['label'], test_size=0.2)

# Train the model
model = MultinomialNB()
model.fit(x_train, y_train)

# Take input from the user
input_text = input("Enter a text: ")

# Pre-process the input text
input_text = re.sub(r'[^a-zA-Z]', ' ', input_text)
input_text = input_text.lower()
input_text = input_text.split()
input_text = [word for word in input_text if word not in stop_words]
input_text = [lemmatizer.lemmatize(word) for word in input_text]
input_text = ' '.join(input_text)

# Vectorize the input text
input_features = vectorizer.transform([input_text])

# Make predictions on the input text
prediction = model.predict(input_features)
confidence = model.predict_proba(input_features)

label = prediction[0]
print("The text is classified as:", "sad" if label == 1 else "not sad")
print("Confidence:", confidence[0][1] if confidence[0][1]>confidence[0][0] else confidence[0][0])

# Ask for user feedback and update the dataset
add = input("Was this right? (y/n) ")
if add.lower() == 'n':
    with open("sad_texts.csv", "a", encoding='utf-8', newline='\n') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC, quotechar="\"")
        writer.writerow([input_text, label])

# Plot for training dataset classes
class_counts = df['label'].value_counts()
# print(class_counts)
plt.bar(['Sad', 'Not Sad'], class_counts)
plt.title('Distribution of Training Dataset Classes')
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.show()

# Plot for confidence scores
print(confidence)
confidence = confidence[0]
print(confidence)
plt.bar(['Not Sad', 'Sad'], confidence)
plt.title('Sadness Detection Confidence')
plt.xlabel('Emotion')
plt.ylabel('Confidence')
plt.show()

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
from wordcloud import WordCloud

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')


def preprocess_text(text):
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    # Convert to lowercase
    text = text.lower()

    # Tokenize the text
    words = text.split()

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Join the words back into a sentence
    text = ' '.join(words)

    return text


def train_model():
    # Load data
    df = pd.read_csv("sad_texts.csv", header=None, names=['text', 'label'])

    # Convert label column to numeric values
    df['label'] = pd.to_numeric(df['label'], errors='coerce')

    # Remove rows with missing label values
    df = df.dropna(subset=['label'])

    # Pre-process text data
    df['text'] = df['text'].apply(preprocess_text)

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2)

    # Vectorize the text data
    vectorizer = CountVectorizer()
    x_train_vectorized = vectorizer.fit_transform(x_train)

    # Train the Naive Bayes classifier
    model = MultinomialNB()
    model.fit(x_train_vectorized, y_train)

    return model, vectorizer, x_test, y_test, df


def predict_sentiment(model, vectorizer, input_text):
    # Pre-process the input text
    input_text = preprocess_text(input_text)

    # Vectorize the input text
    input_features = vectorizer.transform([input_text])

    # Make a prediction
    label = model.predict(input_features)[0]
    confidence = model.predict_proba(input_features)[0]

    return label, confidence


def update_dataset(input_text, label):
    add = input("Was this right? (y/n) ")
    if add.lower() == 'n':
        with open("sad_texts.csv", "a", encoding='utf-8', newline='\n') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC, quotechar="\"")
            writer.writerow([input_text, label])


def plot_class_distribution(df):
    class_counts = df['label'].value_counts()
    plt.bar(['Sad', 'Not Sad'], class_counts)
    plt.title('Distribution of Training Dataset Classes')
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    plt.show()


def plot_sentiment_confidence(confidence):
    plt.bar(['Not Sad', 'Sad'], confidence)
    plt.title('Sadness Detection Confidence')
    plt.xlabel('Emotion')
    plt.ylabel('Confidence')
    plt.show()


def generate_word_clouds(df):
    sad_texts = df[df['label'] == 1]['text']
    not_sad_texts = df[df['label'] == 0]['text']

    sad_wordcloud = WordCloud().generate(' '.join(sad_texts))
    not_sad_wordcloud = WordCloud().generate(' '.join(not_sad_texts))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(sad_wordcloud, interpolation='bilinear')
    plt.title('Words Associated with Sad Sentiment')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(not_sad_wordcloud, interpolation='bilinear')
    plt.title('Words Associated with Not Sad Sentiment')
    plt.axis('off')

    plt.show()


def main():
    # Train the model
    model, vectorizer, x_test, y_test, df = train_model()

    # Take input from the user
    input_text = input("Enter a text: ")

    # Make predictions on the input text
    label, confidence = predict_sentiment(model, vectorizer, input_text)

    # Print the results
    print("The text is classified as:", "sad" if label == 1 else "not sad")
    print("Confidence:", confidence[1] if confidence[1] > confidence[0] else confidence[0])

    # Ask for user feedback and update the dataset
    update_dataset(input_text, label)

    # Plot class distribution
    plot_class_distribution(df)

    # Plot sentiment confidence
    plot_sentiment_confidence(confidence)

    # Generate word clouds
    generate_word_clouds(df)


if __name__ == "__main__":
    main()

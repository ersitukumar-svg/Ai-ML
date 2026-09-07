import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

messages = [
    "Congratulations you won lottery",
    "Get free cashback now",
    "Win a free iPhone today",
    "You have won a cash prize",
    "Claim your free reward",
    "Click here to win money",
    "You are selected for a free gift",
    "Get 5000 rupees instantly",
    "Congratulations you won a prize",
    "Free recharge available now",
    "How are you?",
    "Let's meet tomorrow",
    "Can you call me?",
    "See you at the office",
    "What are you doing?",
    "I will call you later",
    "Let's have lunch together",
    "Are you coming today?",
    "Please send me the document",
    "Meeting is scheduled for tomorrow",
]

labels = [
    "spam",
    "spam",
    "spam",
    "spam",
    "spam",
    "spam",
    "spam",
    "spam",
    "spam",
    "spam",
    "not spam",
    "not spam",
    "not spam",
    "not spam",
    "not spam",
    "not spam",
    "not spam",
    "not spam",
    "not spam",
    "not spam",
]

x = messages
y = labels

print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

vectorizer = CountVectorizer()

X_train_vectorizer = vectorizer.fit_transform(x_train)

model = MultinomialNB()
model.fit(X_train_vectorizer, y_train)

x_test_vertor = vectorizer.transform(x_test)

predictions = model.predict(x_test_vertor)

print("Predictinos", predictions)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

cm = confusion_matrix(y_test, predictions)
print("Confision confusion_matrix", cm)

precision = precision_score(y_test, predictions, pos_label="spam")
print("Precision:", precision)

recall = recall_score(y_test, predictions, pos_label="spam")
print("Recall:", recall)
f1 = f1_score(y_test, predictions, pos_label="spam")
print("F1 Score:", f1)
while True:
    user_message = input("Enter your Message: ")
    if user_message.lower() == "exit":
        print("Program Closed")
        break
    user_message_vector = vectorizer.transform([user_message])
    user_prediction = model.predict(user_message_vector)
    print("Prediction:", user_prediction[0])

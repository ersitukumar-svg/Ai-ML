import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

# 1. Dataset load karo
data = pd.read_csv("../data/spam.csv")
# 2. Input aur output alag karo
x = data["message"]
y = data["label"]
# 3. Dataset ko training aur testing data me divide karo
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
# 4. ML pipeline banao
model = Pipeline([("tfidf", TfidfVectorizer()), ("classifier", MultinomialNB())])
# 5. Model train karo
model.fit(x_train, y_train)
# 6. Test data par prediction
predictions = model.predict(x_test)
# 7. Accuracy calculate karo
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
# 8. Trained model save karo
joblib.dump(model, "../models/spam_model.pkl")
print("Model saved successfully!")

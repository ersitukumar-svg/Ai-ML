import joblib

# Trained model load karo
model = joblib.load("../models/spam_model.pkl")

print("=== AI Spam Detector ===")

while True:
    message = input("\nEnter message (type 'exit' to quit): ")

    if message.lower() == "exit":
        print("Exiting...")
        break

    prediction = model.predict([message])[0]

    if prediction == "spam":
        print("🚨 SPAM")
    else:
        print("✅ HAM")

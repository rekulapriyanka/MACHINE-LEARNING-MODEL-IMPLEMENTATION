import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# ✅ Step 1: Inline CSV Data (no external file needed)
csv_data = """label,message
spam,"Win a free ticket now!"
ham,"Hey, how are you?"
spam,"You have won $1000. Claim now!"
ham,"Are we still meeting today?"
spam,"Congratulations! You won a prize & a free voucher!"
ham,"Can we talk tomorrow?"
"""

# ✅ Step 2: Read the CSV data using StringIO
df = pd.read_csv(StringIO(csv_data))

# ✅ Step 3: Preprocessing (convert labels to binary)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# ✅ Step 4: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.3, random_state=42
)

# ✅ Step 5: Vectorize text data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ✅ Step 6: Train the Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# ✅ Step 7: Make Predictions
y_pred = model.predict(X_test_vec)

# ✅ Step 8: Evaluate the Model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print("\n🧾 Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))


# ✅ Step 9: Test on a Custom Message
def predict_message(msg):
    msg_vec = vectorizer.transform([msg])
    pred = model.predict(msg_vec)[0]
    return "Spam" if pred == 1 else "Not Spam"


# Test with a custom input
custom_msg = "Congratulations! You've won a free trip!"
prediction = predict_message(custom_msg)
print(f"\n🕵️ Prediction for custom message: '{custom_msg}' → {prediction}")

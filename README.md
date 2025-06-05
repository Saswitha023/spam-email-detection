# spam-email-detection
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
mail_data = pd.read_csv('/content/mail_data.csv').fillna('')

# Encode labels: spam = 0, ham = 1
mail_data['Category'] = mail_data['Category'].map({'spam': 0, 'ham': 1})

# Split features and labels
X = mail_data['Message']
Y = mail_data['Category']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Vectorization
vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression()
model.fit(X_train_vec, Y_train)

# Accuracy
train_acc = accuracy_score(Y_train, model.predict(X_train_vec))
test_acc = accuracy_score(Y_test, model.predict(X_test_vec))

print("Training Accuracy:", train_acc)
print("Test Accuracy:", test_acc)

# Function to predict whether the input mail is ham or spam
def predict_mail(input_mail):
    features = vectorizer.transform([input_mail])
    prediction = model.predict(features)[0]
    return "Ham mail" if prediction == 1 else "Spam mail"

# Asking the user to input the email message
user_input = input("Please enter the email message to classify as Ham or Spam: ")

# Get the prediction
result = predict_mail(user_input)
print("Prediction:", result)

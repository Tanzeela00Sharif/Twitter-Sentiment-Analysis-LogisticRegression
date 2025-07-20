# 💬 Twitter Sentiment Analysis using Logistic Regression

This project performs sentiment analysis on tweet-style text data using **Logistic Regression**. The model classifies tweets or reviews into **Positive** or **Negative** categories based on their content.

---

## 📌 Project Features

- ✅ Preprocessing of text data:
  - Lowercasing
  - Removing punctuation
  - Removing stopwords
  - Tokenization
- ✅ Feature extraction using **TF-IDF Vectorizer**
- ✅ Model training using **Logistic Regression**
- ✅ Evaluation using:
  - Accuracy score
  - Confusion matrix
  - Classification report (Precision, Recall, F1-score)
- ✅ Real-time prediction function to check sentiment of new text

---

## 🛠 Tech Stack

- Python
- scikit-learn
- pandas
- NumPy
- nltk (for text preprocessing)
- TfidfVectorizer (for feature extraction)
- matplotlib & seaborn (optional, for visualization)

---

## 📂 Project Structure

sentiment-analysis/
├── sentiment_analysis.py # Core Python script
├── dataset.csv # Dataset used for training (or sample)
├── requirements.txt # Required Python packages
├── README.md # Project documentation
└── example_predictions.txt # Example sentiment predictions

---

## 🧹 Text Preprocessing

- Convert text to lowercase
- Remove URLs, numbers, special characters
- Remove English stopwords using `nltk`
- Use bigram TF-IDF (ngram_range=(1, 2))

---

## 🔍 Model Evaluation Example

Accuracy: 88.5%

Confusion Matrix:
[[4208 369]
[ 621 3477]]

Classification Report:
precision recall f1-score support
Negative 0.87 0.92 0.89 4577
Positive 0.90 0.85 0.88 4098






## 🧪 How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/tanzeela00sarif/twitter-sentiment-analysis.git
   cd twitter-sentiment-analysis

2. Install dependencies
   pip install -r requirements.txt
3. Run the model:

bash
Copy
Edit
python sentiment_analysis.py

4. Predict sentiment for custom text:

python
Copy
Edit
print(predict_sentiment("I love this product!"))

💡 Example Predictions
Review: "you are bad boy" ➡️ Negative
Review: "I love this phone!" ➡️ Positive
Review: "Worst experience ever!" ➡️ Negative


📈 Use Case
This sentiment analysis model can be used in:

Social media monitoring

Customer feedback classification

Product review analysis

Brand reputation tracking

📌 Author
Tanzila Sharif
Data Science Enthusiast | ML | Python | Prompt Engineering
GitHub Profile

📃 License
This project is open-source and free to use under the MIT License.

# spam_classifier.py
# 📬 Spam Email/SMS Classifier using scikit-learn (TF-IDF + Logistic Regression)

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

# ------------------------------------------------
# 1) Load Dataset (or create sample)
# ------------------------------------------------
csv_path = "sample_sms.csv"

if not os.path.exists(csv_path):
    sample = [
        ("Win a brand new iPhone! Click the link now", "spam"),
        ("Exclusive offer!!! Claim your prize today", "spam"),
        ("Your loan is approved. Call now", "spam"),
        ("Lowest insurance rates available. Reply YES", "spam"),
        ("Reminder: Your appointment is tomorrow at 10am", "ham"),
        ("Can we meet at 5 pm for project work?", "ham"),
        ("Mom: please buy milk on your way home", "ham"),
        ("Your OTP is 392014. Do not share it.", "ham"),
        ("Limited time sale, buy 1 get 1 free!!!", "spam"),
        ("Meeting rescheduled to Monday", "ham"),
        ("URGENT! You won a lottery worth $5000", "spam"),
        ("Let's catch up for lunch?", "ham"),
        ("Congratulations, you've been selected! Act now", "spam"),
        ("Flight booking confirmed for 7 PM", "ham"),
        ("New features launched—update your app today", "ham"),
    ]
    pd.DataFrame(sample, columns=["text", "label"]).to_csv(csv_path, index=False)

df = pd.read_csv(csv_path)
print("Dataset shape:", df.shape)
print(df.head())

# ------------------------------------------------
# 2) Train/Test Split
# ------------------------------------------------
X = df['text'].astype(str)
y = df['label'].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ------------------------------------------------
# 3) Pipeline: TF-IDF + Logistic Regression
# ------------------------------------------------
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=1000))
])

# ------------------------------------------------
# 4) Train Baseline Model
# ------------------------------------------------
pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)

# ------------------------------------------------
# 5) Evaluation Function
# ------------------------------------------------
def show_metrics(y_true, y_pred, y_prob=None, title="Results"):
    print("\n" + "="*50)
    print(title)
    print("="*50)
    print("Accuracy :", f"{accuracy_score(y_true, y_pred):.3f}")
    print("Precision:", f"{precision_score(y_true, y_pred, pos_label='spam'):.3f}")
    print("Recall   :", f"{recall_score(y_true, y_pred, pos_label='spam'):.3f}")
    print("F1 Score :", f"{f1_score(y_true, y_pred, pos_label='spam'):.3f}")
    print("\nClassification Report:\n", classification_report(y_true, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=['ham', 'spam'])
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.xticks([0, 1], ['ham', 'spam'])
    plt.yticks([0, 1], ['ham', 'spam'])
    for (i, j), z in np.ndenumerate(cm):
        plt.text(j, i, str(z), ha='center', va='center')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # ROC Curve
    if y_prob is not None:
        fpr, tpr, _ = roc_curve((y_true == 'spam').astype(int), y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.show()

# ------------------------------------------------
# 6) Show Baseline Metrics
# ------------------------------------------------
y_prob = pipeline.predict_proba(X_test)[:, 1]
show_metrics(y_test, preds, y_prob, title="Baseline Model")

# ------------------------------------------------
# 7) Cross-validation
# ------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring='f1_weighted')
print("\n5-Fold CV F1 Scores:", scores)
print("Mean ± Std:", f"{scores.mean():.3f} ± {scores.std():.3f}")

# ------------------------------------------------
# 8) Hyperparameter Tuning (Grid Search)
# ------------------------------------------------
param_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [1, 2, 3],
    "clf__C": [0.5, 1.0, 2.0, 4.0],
    "clf__penalty": ["l2"],
    "clf__solver": ["lbfgs"]
}

grid = GridSearchCV(pipeline, param_grid, scoring="f1_weighted", cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

print("\nBest Parameters:", grid.best_params_)
best_model = grid.best_estimator_

# ------------------------------------------------
# 9) Evaluate Tuned Model
# ------------------------------------------------
preds_best = best_model.predict(X_test)
y_prob_best = best_model.predict_proba(X_test)[:, 1]
show_metrics(y_test, preds_best, y_prob_best, title="Tuned Model")

# ------------------------------------------------
# 10) Save the Model
# ------------------------------------------------
os.makedirs("artifacts", exist_ok=True)
model_path = os.path.join("artifacts", "spam_classifier.pkl")
joblib.dump(best_model, model_path)
print("\n✅ Model saved at:", model_path)

# ------------------------------------------------
# 11) Inference on New Messages
# ------------------------------------------------
examples = [
    "Congratulations! You've won a free vacation. Call now to claim.",
    "Team, the meeting is moved to 3 PM. See you.",
    "Lowest insurance rates guaranteed—apply today!",
    "Your OTP is 123456. Do not share it with anyone."
]

loaded = joblib.load(model_path)
preds = loaded.predict(examples)

print("\n🔍 Predictions on new messages:")
for text, label in zip(examples, preds):
    print(f"[{label.upper()}] {text}")

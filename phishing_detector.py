import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# Load the phishing dataset (from Kaggle)
def run_detector():
    print("Loading phishing dataset...")

    data = pd.read_csv('phishing.csv')

    # Separate the different features + target labels
    raw_features = data.drop('class', axis=1)
    labels = data['class']

    # Scale the features to illuminate accurate threat detection
    scaler = StandardScaler()
    features = scaler.fit_transform(raw_features)

    # Divide the dataset into training + testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    return train_features, test_features, train_labels, test_labels

# Teach the AI how to detect phishing emails (Logistic Regression + Random Forest)
def train_phish_detector(train_features, train_labels):
    print(" Training base models...")

    logistic_model = LogisticRegression(max_iter=3000)
    randfor_model = RandomForestClassifier()

    # Merge both algorithms into one ensemble model for smart detection
    phish_detector_ai = VotingClassifier(estimators=[
        ('Logistic Regression', logistic_model),
        ('Random Forest', randfor_model)
    ], voting='soft')

    phish_detector_ai.fit(train_features, train_labels)
    return phish_detector_ai

# Evaluate the AI model: confusion matrix, performance report, and visuals
def performance_report(model, test_features, test_labels):
    print(" Evaluating AI phishing detector...\n")

    phish_predictions = model.predict(test_features)
    detection_accuracy = model.score(test_features, test_labels)
    print(f" Final Ensemble Model Accuracy: {detection_accuracy * 100:.2f}%")

    # Create the formatted confusion matrix using pandas
    labels_display = ['Phishing (-1)', 'Legit (1)']
    conf_matrix = confusion_matrix(test_labels, phish_predictions)
    df_cm = pd.DataFrame(conf_matrix, index=labels_display, columns=labels_display)

    print("\n Formatted Confusion Matrix:")
    print(df_cm)

    # Print raw confusion matrix just for reference
    print("\nConfusion Matrix (Raw):")
    print(conf_matrix)

    # Graphical pink confusion matrix 
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=labels_display)
    disp.plot(cmap='PuRd')  
    plt.title("Phishing Detection Confusion Matrix (Pink Edition)")
    # Matrix as a PNG
    plt.savefig("phishing_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Print the classification report
    print("\n Classification Report:")
    print(classification_report(test_labels, phish_predictions))

    print("\n Detection complete. Model is ready for deployment!")

# Run the full pipeline
if __name__ == "__main__":
    train_X, test_X, train_y, test_y = run_detector()
    ai_model = train_phish_detector(train_X, train_y)
    performance_report(ai_model, test_X, test_y)

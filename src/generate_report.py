import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Load metrics from JSON file
def load_metrics(filename='metrics.json'):
    with open(filename, 'r') as f:
        return json.load(f)

# Function to generate classification report
def generate_classification_report(true_labels, predicted_labels, target_names):
    report = classification_report(true_labels, predicted_labels, target_names=target_names)
    print("Classification Report:")
    print(report)
    return report

# Function to plot confusion matrix
def plot_confusion_matrix(true_labels, predicted_labels, target_names):
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    plt.show()

# Function to plot accuracy and loss curves
def plot_training_curves(train_acc, val_acc, train_loss, val_loss):
    epochs = range(1, len(train_acc) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.show()

# Function to save classification report to a text file
def save_report_to_file(report, filename='classification_report.txt'):
    with open(filename, 'w') as f:
        f.write(report)
    print(f'Report saved to {filename}')

# Function to create a PDF report
def create_pdf_report(filename='full_report.pdf', report_text=""):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    c.drawString(100, height - 50, "Model Classification Report")
    text_object = c.beginText(100, height - 100)
    text_object.textLines(report_text)

    c.drawText(text_object)
    c.showPage()
    c.drawImage("confusion_matrix.png", 50, 300, width=500, height=300)
    c.drawImage("training_curves.png", 50, 50, width=500, height=200)
    c.save()
    print(f'Report saved as {filename}')

# Main function to generate the complete report
def generate_full_report(target_names):
    # Load metrics
    metrics = load_metrics()

    # Load predictions
    with open('predictions.json', 'r') as f:
        predictions = json.load(f)

    true_labels = predictions['true_labels']
    predicted_labels = predictions['predicted_labels']

    # Generate classification report
    report = generate_classification_report(true_labels, predicted_labels, target_names)

    # Plot and save confusion matrix
    plot_confusion_matrix(true_labels, predicted_labels, target_names)

    # Plot and save training curves
    plot_training_curves(metrics['train_acc'], metrics['val_acc'], metrics['train_loss'], metrics['val_loss'])

    # Save the classification report to a text file
    save_report_to_file(report, filename='classification_report.txt')

    # Create a PDF report
    create_pdf_report(filename='full_report.pdf', report_text=report)

if __name__ == "__main__":
    target_names = ['garbage', 'limestone', 'sandstone', 'shale']
    generate_full_report(target_names)

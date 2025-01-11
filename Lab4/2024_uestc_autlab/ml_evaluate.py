import pickle
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns
from dataset import MyCOCODataset

classes = 7  # Number of classes in the dataset

def plot_confusion_matrix(matrix, class_names, output_path):
    """
    Plot and save the confusion matrix.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Decision Tree Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(output_path)
    plt.close()


def evaluate_model(predicted, ground_truth, class_names):
    """
    Evaluate the model using various metrics and plot the confusion matrix.
    """
    # Print classification report
    print("Classification Report:")
    report = classification_report(
        ground_truth, 
        predicted, 
        target_names=class_names, 
        digits=4, 
        zero_division=0
    )
    print(report)

    # Compute and print confusion matrix
    matrix = confusion_matrix(ground_truth, predicted, labels=range(len(class_names)))
    print("Confusion Matrix:")
    print(matrix)

    # Plot confusion matrix
    plot_confusion_matrix(matrix, class_names, './Lab4/image/confusion_matrix_decision_tree.png')


def main():
    # Load trained decision tree model
    with open('./Lab4/2024_uestc_autlab/outputs/model.pkl', 'rb') as file:
        model = pickle.load(file)

    # Load test dataset
    dataset = MyCOCODataset(
        "./Lab4/coco",
        "./Project2/2024_uestc_autlab/data/data_coco_test/annotations.json",
        output_size=(128, 128),
    )

    # Prepare ground truth and predictions
    test_predicted = []
    test_ground_truth = []
    for i in range(len(dataset)):
        img, label = dataset[i]
        img = np.expand_dims(img, axis=0)  # Add batch dimension for single image
        prediction = model.forward(img)[0]  # Use the forward method for prediction
        test_predicted.append(prediction)
        test_ground_truth.append(label)

    # Convert to numpy arrays
    test_predicted = np.array(test_predicted)
    test_ground_truth = np.array(test_ground_truth)

    # Class names (replace with actual class labels if available)
    class_names = [f"Class {i}" for i in range(classes)]

    # Evaluate model
    evaluate_model(test_predicted, test_ground_truth, class_names)


if __name__ == "__main__":
    main()
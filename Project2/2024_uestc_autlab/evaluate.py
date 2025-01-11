# evaluate.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from dataset import MyCOCODataset
from model import AlexNet

def normalize_batch(images, device):
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    images = images.float() / 255.0
    images = images.permute(0, 3, 1, 2)
    images = (images - MEAN.to(device)) / STD.to(device)
    return images

def evaluate_model(model_path, test_loader, device, num_classes=7):
    # Load model
    model = AlexNet(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Collect predictions
    predictions = []
    ground_truth = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = normalize_batch(images, device)
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
            ground_truth.extend(labels.numpy())
    
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    # Calculate metrics
    accuracy = (predictions == ground_truth).mean()
    cm = confusion_matrix(ground_truth, predictions)
    cr = classification_report(ground_truth, predictions)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('CNN Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('./Project2/image/confusion_matrix_final.png')
    plt.close()
    
    return accuracy, cm, cr

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load test dataset
    test_set = MyCOCODataset(
        "./Lab4/coco",
        "./Project2/2024_uestc_autlab/data/data_coco_test/annotations.json",
    )
    test_loader = DataLoader(test_set, batch_size=32)
    
    # Evaluate
    accuracy, conf_matrix, class_report = evaluate_model('./Project2/2024_uestc_autlab/outputs/best_model3.pth', test_loader, device)
    print(f"Final Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(class_report)
    print("\nConfusion Matrix:")
    print(conf_matrix)
# train.py
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MyCOCODataset
from model import AlexNet

# Hyperparameters
LEARNING_RATE = 3e-4
BATCH_SIZE = 32
NUM_CLASSES = 7
NUM_EPOCHS = 50
PATIENCE = 8
WEIGHT_DECAY = 1e-4

# Data normalization values
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def normalize_batch(images, device):
    images = images.float() / 255.0
    images = images.permute(0, 3, 1, 2)
    images = (images - MEAN.to(device)) / STD.to(device)
    return images

def train_model(model, train_loader, val_loader, device):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=3, verbose=True
    )
    
    best_accuracy = 0
    no_improvement = 0
    
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images = normalize_batch(images, device)
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = normalize_batch(images, device)
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] - '
              f'Loss: {avg_train_loss:.4f} - '
              f'Validation Accuracy: {accuracy:.4f}')
        
        scheduler.step(accuracy)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            no_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
            }, './Project2/2024_uestc_autlab/outputs/best_model.pth')
        else:
            no_improvement += 1
        
        if no_improvement >= PATIENCE:
            print(f'Early stopping triggered after epoch {epoch+1}')
            break

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_set = MyCOCODataset(
        "./Lab4/coco",
        "./Project2/2024_uestc_autlab/data/data_coco_train/annotations.json",
    )
    val_set = MyCOCODataset(
        "./Lab4/coco",
        "./Project2/2024_uestc_autlab/data/data_coco_valid/annotations.json",
    )
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    
    model = AlexNet(num_classes=NUM_CLASSES)
    train_model(model, train_loader, val_loader, device)
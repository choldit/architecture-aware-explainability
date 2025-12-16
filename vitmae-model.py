import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import random
from transformers import ViTMAEModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

#    git clone git@github.com:Alsalivan/ecgcmr.git
#     cd ecgcmr/external/transformers
#     pip install -e 

# Dataset with proper ECG formatting
class NpyDataset(Dataset):
    def __init__(self, root_dir, transform=None, multiplying_factors=None):
        """
        Args:
            root_dir (string): Directory with all the classes
            transform (callable, optional): Optional transform to be applied
            multiplying_factors (dict): Dictionary mapping class names to multiplication factors
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            for file in class_dir.glob("*.npy"):
                self.samples.append((file, self.class_to_idx[class_name]))
        
        if multiplying_factors:
            self.oversample_classes(multiplying_factors)
        
        self.targets = [label for _, label in self.samples]

    def oversample_classes(self, multiplying_factors):
        """Oversample classes based on multiplying factors"""
        class_samples = {}
        for file_path, label in self.samples:
            class_name = self.idx_to_class[label]
            if class_name not in class_samples:
                class_samples[class_name] = []
            class_samples[class_name].append((file_path, label))
        
        new_samples = []
        for class_name, samples in class_samples.items():
            factor = multiplying_factors.get(class_name, 1)
            new_samples.extend(samples)
            if factor > 1:
                for _ in range(factor - 1):
                    oversampled = random.choices(samples, k=len(samples))
                    new_samples.extend(oversampled)
        
        self.samples = new_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        array = np.load(file_path)  # shape (1280,)
        
        # Reshape to (1, H, W) - treating as single-channel image
        array = array.reshape(1, 32, 40)
        
        # Normalize
        array = (array - array.min()) / (array.max() - array.min() + 1e-8)
        
        # Convert to 3 channels (repeat grayscale) for ViT-MAE compatibility
        array = np.repeat(array, 3, axis=0)  # (3, 32, 40)
        
        tensor = torch.tensor(array, dtype=torch.float)
        
        if self.transform:
            tensor = self.transform(tensor)
            
        return tensor, label

    def get_class_distribution(self):
        """Return the distribution of classes in the dataset"""
        return Counter([self.idx_to_class[label] for label in self.targets])


# ECG ViT-MAE Classifier
class ECGViTMAEClassifier(nn.Module):
    def __init__(self, num_classes=2, img_size=(32, 40), freeze_encoder=False, dropout=0.1):
        """
        Args:
            num_classes: Number of output classes
            img_size: Input image size (H, W)
            freeze_encoder: If True, freeze the pretrained encoder weights
            dropout: Dropout rate in classification head
        """
        super().__init__()
        
        # Load the ECG-pretrained ViT-MAE model
        print("Loading ECG-pretrained ViT-MAE model from alsalivan/vitmae-ecg...")
        try:
            self.vit_mae = ViTMAEModel.from_pretrained(
                ".ecgmr/vitmae-ecg",
                image_size=img_size,
                ignore_mismatched_sizes=True  # Handle size differences
            )
            print("Successfully loaded pretrained model!")
        except Exception as e:
            print(f"Error loading pretrained model: {e}")
            print("Make sure you have installed the forked transformers library:")
            print("git clone git@github.com:Alsalivan/ecgcmr.git")
            print("cd ecgcmr/external/transformers")
            print("pip install -e .")
            raise
        
        # Freeze encoder if specified
        if freeze_encoder:
            print("Freezing encoder weights - only training classifier head")
            for param in self.vit_mae.parameters():
                param.requires_grad = False
        else:
            print("Fine-tuning entire model")
        
        # Get hidden size from config
        hidden_size = self.vit_mae.config.hidden_size
        
        # Classification head with dropout
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # Initialize classification head
        self._init_classifier()
    
    def _init_classifier(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            logits: Output tensor of shape (B, num_classes)
        """
        # Get ViT-MAE encoder outputs
        outputs = self.vit_mae(pixel_values=x)
        
        # Use CLS token (first token) for classification
        cls_output = outputs.last_hidden_state[:, 0]
        
        # Pass through classification head
        logits = self.classifier(cls_output)
        
        return logits
    
    def get_attention_maps(self, x):
        """Extract attention maps for visualization"""
        outputs = self.vit_mae(pixel_values=x, output_attentions=True)
        return outputs.attentions


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=1e-4, weight_decay=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # Use different learning rates for encoder and classifier
    encoder_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'classifier' in name:
                classifier_params.append(param)
            else:
                encoder_params.append(param)
    
    # Classifier gets higher learning rate, encoder gets lower (if unfrozen)
    optimizer = AdamW([
        {'params': encoder_params, 'lr': learning_rate * 0.1},  # 10x smaller for encoder
        {'params': classifier_params, 'lr': learning_rate}
    ], weight_decay=weight_decay)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    early_stopping = EarlyStopping(patience=10, verbose=True, path='best_ecg_vitmae_model.pth')
    
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        scheduler.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        print('-' * 50)
        
        # Early stopping check
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    # Load the best model
    model.load_state_dict(torch.load('best_ecg_vitmae_model.pth'))
    
    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }


def evaluate_model(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            probs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_targets), np.array(all_probs)


def plot_training_curves(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curve
    ax1.plot(history['train_losses'], label='Train Loss', linewidth=2)
    ax1.plot(history['val_losses'], label='Val Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curve
    ax2.plot(history['train_accs'], label='Train Accuracy', linewidth=2)
    ax2.plot(history['val_accs'], label='Val Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()


# Main execution
if __name__ == "__main__":
    print("=" * 70)
    print("ECG ViT-MAE for Aortic Stenosis Classification")
    print("=" * 70)
    
 
    # Define multiplying factors based on the paper
    multiplying_factors = {
        "No_AS": 5,    # ×5 for no aortic stenosis
        "AS": 21       # ×21 for severe AS
    }

    # Create datasets
    print("\nLoading datasets...")
    train_dataset = NpyDataset(
        root_dir="/70_20_10/train",
        multiplying_factors=multiplying_factors
    )

    val_dataset = NpyDataset(
        root_dir="/70_20_10/val", 
        multiplying_factors=multiplying_factors
    )

    test_dataset = NpyDataset(
        root_dir="/70_20_10/test"
    )

    # Check class distribution
    print("\n" + "=" * 50)
    print("Dataset Statistics:")
    print("=" * 50)
    print("\nTrain set - After oversampling:")
    print(train_dataset.get_class_distribution())
    print(f"Total samples: {len(train_dataset)}")
    
    print("\nValidation set - After oversampling:")
    print(val_dataset.get_class_distribution())
    print(f"Total samples: {len(val_dataset)}")
    
    print("\nTest set - Original distribution:")
    print(test_dataset.get_class_distribution())
    print(f"Total samples: {len(test_dataset)}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    # Initialize ECG ViT-MAE model
    print("\n" + "=" * 50)
    print("Model Initialization:")
    print("=" * 50)
    
    model = ECGViTMAEClassifier(
        num_classes=2,
        img_size=(32, 40),
        freeze_encoder=False,  # Set to True to only train classifier
        dropout=0.1
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train the model
    print("\n" + "=" * 50)
    print("Starting Training:")
    print("=" * 50)
    
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=100,
        learning_rate=1e-4,
        weight_decay=1e-4
    )
    
    # Plot training curves
    plot_training_curves(history)
    
    # Evaluate on test set
    print("\n" + "=" * 50)
    print("Test Set Evaluation:")
    print("=" * 50)
    
    test_preds, test_targets, test_probs = evaluate_model(trained_model, test_loader)
    
    # Calculate primary metrics
    accuracy = accuracy_score(test_targets, test_preds)
    precision = precision_score(test_targets, test_preds, average='weighted')
    recall = recall_score(test_targets, test_preds, average='weighted')
    f1 = f1_score(test_targets, test_preds, average='weighted')
    
    # Per-class metrics
    precision_per_class = precision_score(test_targets, test_preds, average=None)
    recall_per_class = recall_score(test_targets, test_preds, average=None)
    f1_per_class = f1_score(test_targets, test_preds, average=None)
    
    print("\n" + "=" * 50)
    print("OVERALL METRICS")
    print("=" * 50)
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")
    
    print("\n" + "=" * 50)
    print("PER-CLASS METRICS")
    print("=" * 50)
    print(f"\n{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 50)
    print(f"{'No_AS':<15} {precision_per_class[0]*100:>10.2f}%  {recall_per_class[0]*100:>10.2f}%  {f1_per_class[0]*100:>10.2f}%")
    print(f"{'AS':<15} {precision_per_class[1]*100:>10.2f}%  {recall_per_class[1]*100:>10.2f}%  {f1_per_class[1]*100:>10.2f}%")
    
    # Confusion matrix
    cm = confusion_matrix(test_targets, test_preds)
    print("\n" + "=" * 50)
    print("CONFUSION MATRIX")
    print("=" * 50)
    print(f"\n{'':>15} {'Predicted No_AS':>18} {'Predicted AS':>15}")
    print(f"{'Actual No_AS':<15} {cm[0][0]:>18} {cm[0][1]:>15}")
    print(f"{'Actual AS':<15} {cm[1][0]:>18} {cm[1][1]:>15}")
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)



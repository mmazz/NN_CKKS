import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from pathlib import Path

class MNISTCSV(Dataset):
    def __init__(self, csv_path):
        data = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)

        # Primera columna: labels
        self.labels = torch.tensor(data[:, 0], dtype=torch.long)

        # Resto de columnas: píxeles (784)
        # Normalizamos a [-1,1] porque ya no usamos ToTensor()
        self.images = torch.tensor((data[:, 1:] / 255.0) * 2.0 - 1.0,
                                   dtype=torch.float32)
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
dir = Path("data/weights")

# Crear la carpeta si no existe
dir.mkdir(exist_ok=True)

# --- 1. Dataset ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # Flatten 28x28 → 784
])

train_dataset = MNISTCSV("data/mnist_train.csv")
test_dataset  = MNISTCSV("data/mnist_test.csv")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

# --- 2. Modelo con activación x^2 ---
class HomomorphicMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = 0.98*x - 0.23*x**3              # ACTIVACIÓN x²  ←  clave para CKKS
        x = self.fc2(x)
        return x                # Output logits (Softmax lo hacemos en loss)

model = HomomorphicMLP()


# --- 3. Entrenamiento ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")


# --- 4. Evaluación ---
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        preds = torch.argmax(output, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

print(f"Accuracy: {correct/total*100:.2f}%")


W1 = model.fc1.weight.detach().cpu().numpy()   # (64, 784)
b1 = model.fc1.bias.detach().cpu().numpy()     # (64,)

W2 = model.fc2.weight.detach().cpu().numpy()   # (10, 64)
b2 = model.fc2.bias.detach().cpu().numpy()     # (10,)

# Guardar como CSV
np.savetxt("data/weights/W1.csv", W1, delimiter=",")
np.savetxt("data/weights/b1.csv", b1, delimiter=",")
np.savetxt("data/weights/W2.csv", W2, delimiter=",")
np.savetxt("data/weights/b2.csv", b2, delimiter=",")

print("Listo: guardado en W1.csv, b1.csv, W2.csv, b2.csv")

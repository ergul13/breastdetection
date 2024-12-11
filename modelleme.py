import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import xml.etree.ElementTree as ET
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
import random
import matplotlib.patches as patches


class MammoDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        self.transform = transform
        self.label_map = {"meme": 1, "breast": 1, "background": 0}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        annotation_path = os.path.join(self.annotation_dir, self.image_files[idx].replace('.png', '.xml'))

        image = Image.open(image_path).convert('L')
        if self.transform:
            image = self.transform(image)

        tree = ET.parse(annotation_path)
        root = tree.getroot()
        label_name = root.find('object').find('name').text

        if label_name not in self.label_map:
            raise ValueError(f"Etiket tanınmıyor: {label_name}")
        label = self.label_map[label_name]

        return image, label


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

batch_size = 32

train_dataset = MammoDataset(image_dir=r'C:\Users\canla\Desktop\ayrilmis_veriler\train',
                             annotation_dir=r'C:\Users\canla\Desktop\ayrilmis_veriler\train',
                             transform=transform)
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

test_dataset = MammoDataset(image_dir=r'C:\Users\canla\Desktop\ayrilmis_veriler\test',
                            annotation_dir=r'C:\Users\canla\Desktop\ayrilmis_veriler\test',
                            transform=transform)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

val_dataset = MammoDataset(image_dir=r'C:\Users\canla\Desktop\ayrilmis_veriler\validation',
                           annotation_dir=r'C:\Users\canla\Desktop\ayrilmis_veriler\validation',
                           transform=transform)
valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


PATH = r'C:\Users\canla\Desktop\netModel\model.pth'
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

train_again = input("Modeli tekrar eğitmek istiyor musunuz? (Y/N): ").strip().lower()

if train_again == 'y':
    print("Model eğitiliyor...")
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 10 == 0:
                print(f"Epoch [{epoch + 1}/2], Step [{i + 1}/{len(trainloader)}], Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch + 1}/2] tamamlandı. Ortalama Loss: {running_loss / len(trainloader):.4f}")

    torch.save(net.state_dict(), PATH)
    print(f"Model ağırlıkları kaydedildi.")
else:
    try:
        net.load_state_dict(torch.load(PATH))
        net.eval()
        print("Model ağırlıkları yüklendi. Test aşamasına geçiliyor...")
    except Exception as e:
        print(f"Model ağırlıkları yüklenirken bir hata oluştu: {e}")
        print("Lütfen modeli yeniden eğitin.")

val_loss = 0.0
correct = 0
total = 0
all_labels = []
all_predictions = []

with torch.no_grad():
    for i, data in enumerate(valloader, 0):
        inputs, labels = data
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        print(f"Step [{i + 1}/{len(valloader)}], Validation Loss: {loss.item():.4f}")
        val_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

    avg_val_loss = val_loss / len(valloader)
    val_accuracy = 100 * correct / total
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%, F1 Score: {f1:.2f}")


def draw_bounding_boxes(img, label, coords):
    img = img / 2 + 0.5
    npimg = img.numpy()
    fig, ax = plt.subplots(1)
    ax.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    x_min, y_min, x_max, y_max = coords
    rect = patches.Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        linewidth=2,
        edgecolor='r',
        facecolor='none'
    )
    ax.add_patch(rect)
    plt.title(f"Label: {label}")
    plt.show()


image_files = [f for f in os.listdir(r'C:\Users\canla\Desktop\ayrilmis_veriler\test') if f.endswith('.png')]
random_files = random.sample(image_files, 10)

for image_file in random_files:
    annotation_path = os.path.join(r'C:\Users\canla\Desktop\ayrilmis_veriler\test', image_file.replace('.png', '.xml'))

    if not os.path.exists(annotation_path):
        print(f"Annotation file not found for {image_file}")
        continue

    img_path = os.path.join(r'C:\Users\canla\Desktop\ayrilmis_veriler\test', image_file)
    img = Image.open(img_path).convert('L')
    img_original = img.copy()
    img = transform(img)

    tree = ET.parse(annotation_path)
    root = tree.getroot()
    bndbox = root.find('object').find('bndbox')
    x_min = int(bndbox.find('xmin').text)
    y_min = int(bndbox.find('ymin').text)
    x_max = int(bndbox.find('xmax').text)
    y_max = int(bndbox.find('ymax').text)

    original_width, original_height = img_original.size
    x_min_scaled = x_min * (224 / original_width)
    y_min_scaled = y_min * (224 / original_height)
    x_max_scaled = x_max * (224 / original_width)
    y_max_scaled = y_max * (224 / original_height)

    draw_bounding_boxes(img, 1, (x_min_scaled, y_min_scaled, x_max_scaled, y_max_scaled))

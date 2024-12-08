import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import xml.etree.ElementTree as ET
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np

# 1. Özel Dataset Sınıfı
class MammoDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        self.transform = transform
        self.label_map = {"meme": 0, "breast": 0, "non-meme": 1, "non-breast": 1}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Görüntü ve XML yolları
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        annotation_path = os.path.join(self.annotation_dir, self.image_files[idx].replace('.png', '.xml'))

        # Görüntüyü yükle
        image = Image.open(image_path).convert('RGB')  # EfficientNet RGB giriş bekler
        if self.transform:
            image = self.transform(image)

        # XML'den etiket okuma
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        label_name = root.find('object').find('name').text

        # Etiket haritalama
        if label_name not in self.label_map:
            raise ValueError(f"Etiket tanınmıyor: {label_name}")
        label = self.label_map[label_name]

        return image, label

# 2. Veri Dönüşümleri
transform = transforms.Compose([
    transforms.Resize((456, 456)),  # EfficientNet-B5 için giriş boyutu
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalizasyonu
])

batch_size = 16

# Dataset ve DataLoader
train_dataset = MammoDataset(
    image_dir=r'C:\Users\canla\Desktop\ayrilmis_veriler\train',
    annotation_dir=r'C:\Users\canla\Desktop\ayrilmis_veriler\train',
    transform=transform
)

test_dataset = MammoDataset(
    image_dir=r'C:\Users\canla\Desktop\ayrilmis_veriler\test',
    annotation_dir=r'C:\Users\canla\Desktop\ayrilmis_veriler\test',
    transform=transform
)

trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# 3. EfficientNet-B5 Modelini Yükle
efficientnet_b5 = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.IMAGENET1K_V1)  # Pretrained model
efficientnet_b5.classifier[1] = nn.Linear(efficientnet_b5.classifier[1].in_features, 2)  # Çıkış sınıfını 2'ye uyarlayın

# 4. Kayıp Fonksiyonu ve Optimizasyon
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(efficientnet_b5.parameters(), lr=0.0001)

# 5. Model Eğitimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
efficientnet_b5.to(device)

for epoch in range(5):  # 5 epoch
    efficientnet_b5.train()  # Modeli eğitim moduna al
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Grad'leri sıfırla
        optimizer.zero_grad()

        # İleri + geri + optimize
        outputs = efficientnet_b5(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Kayıpları yazdır
        running_loss += loss.item()
        if i % 10 == 9:  # Her 10 minibatch'te bir yazdır
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 10:.3f}")
            running_loss = 0.0

print("Eğitim Tamamlandı")

# 6. Modeli Kaydet
PATH = r'C:\Users\canla\Desktop\netModel\efficientnet_b5_model.pth'
torch.save(efficientnet_b5.state_dict(), PATH)

# 7. Test
efficientnet_b5.eval()  # Modeli değerlendirme moduna al
correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = efficientnet_b5(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Doğruluğu: {100 * correct / total:.2f}%")

# 8. Test Edilen Görselleri Göster
def imshow(img):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(testloader)
images, labels = next(dataiter)
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{labels[j]}' for j in range(len(labels))))

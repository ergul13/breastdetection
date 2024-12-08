import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import xml.etree.ElementTree as ET
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# 1. Özel Dataset Sınıfı
class MammoDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        self.transform = transform
        # Etiket haritası
        self.label_map = {"meme": 0, "breast": 0, "non-meme": 1, "non-breast": 1}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Görüntü ve XML yolları
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        annotation_path = os.path.join(self.annotation_dir, self.image_files[idx].replace('.png', '.xml'))

        # Görüntüyü yükle
        image = Image.open(image_path).convert('L')  # Gri tonlama
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
    transforms.Resize((32, 32)),  # Görüntü boyutunu küçült
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize
])

batch_size = 4

# 3. Veri Yükleyiciler
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

# 4. İkili Sınıflandırma İçin CNN Modeli
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # Gri tonlama için 1 kanal
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)  # İkili sınıflandırma için çıkış 2

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # Düzleştir
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 5. Modelin Yüklenmesi
PATH = r'C:\Users\canla\Desktop\netModel'  # Daha önce kaydedilen model dosyası
net = Net()
net.load_state_dict(torch.load(PATH))
net.eval()  # Modeli test moduna geçir

print("Model başarıyla yüklendi ve test moduna alındı.")

# Test süreci
correct = 0
total = 0
all_predictions = []
all_labels = []

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Tahminleri ve gerçek etiketleri kaydet
        all_predictions.extend(predicted.numpy())
        all_labels.extend(labels.numpy())

print(f'Test Doğruluğu: {100 * correct / total:.2f}%')

# Test sonuçlarının detaylı analizi
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix

# Test setindeki sınıf dağılımını kontrol et
print("\nTest setindeki sınıf dağılımı:")
print(Counter(all_labels))

# Test setinde sadece bir sınıf varsa, kullanıcıyı uyar
if len(set(all_labels)) < 2:
    print("UYARI: Test setinde yalnızca bir sınıf mevcut. Classification report ve confusion matrix oluşturulamıyor.")
else:
    # Sınıf raporu ve karmaşıklık matrisi oluştur
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=["meme", "non-meme"], labels=[0, 1]))

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_predictions))



# 7. Test Edilen Görselleri Göster
def imshow(img):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(testloader)
images, labels = next(dataiter)
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{labels[j]}' for j in range(len(labels))))

# 8. Sonuç Analizi
from sklearn.metrics import classification_report, confusion_matrix

print("\nClassification Report:")
print(classification_report(all_labels, all_predictions, target_names=["meme", "non-meme"]))

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_predictions))


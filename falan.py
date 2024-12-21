import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import xml.etree.ElementTree as ET


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

png_dir = r"C:\Users\canla\Desktop\donusum.png"
xml_dir = r"C:\Users\canla\Desktop\donusum.xml"

png_files = [f for f in os.listdir(png_dir)]
xml_files = [f for f in os.listdir(xml_dir)]

matched_files = [(png, png.replace('.png', '.xml')) for png in png_files if png.replace('.png', '.xml') in xml_files]

from sklearn.model_selection import train_test_split
train_files, test_files = train_test_split(matched_files, test_size=0.2, random_state=42)

def calculate_iou(pred_box, true_box):
    pred_xmin, pred_ymin, pred_xmax, pred_ymax = pred_box
    true_xmin, true_ymin, true_xmax, true_ymax = true_box

    inter_xmin = max(pred_xmin, true_xmin)
    inter_ymin = max(pred_ymin, true_ymin)
    inter_xmax = min(pred_xmax, true_xmax)
    inter_ymax = min(pred_ymax, true_ymax)

    inter_width = max(0, inter_xmax - inter_xmin)
    inter_height = max(0, inter_ymax - inter_ymin)
    inter_area = inter_width * inter_height

    pred_area = (pred_xmax - pred_xmin) * (pred_ymax - pred_ymin)
    true_area = (true_xmax - true_xmin) * (true_ymax - true_ymin)

    union_area = pred_area + true_area - inter_area

    iou = inter_area / union_area if union_area != 0 else 0
    return iou

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, files, png_dir, xml_dir, transform=None):
        self.files = files
        self.png_dir = png_dir
        self.xml_dir = xml_dir
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        png_file, xml_file = self.files[idx]

        image_path = os.path.join(self.png_dir, png_file)
        image = torchvision.io.read_image(image_path, mode=torchvision.io.image.ImageReadMode.GRAY)
        image = image.float() / 255.0
        original_size = image.shape[1:]

        if self.transform:
            image = self.transform(image)

        xml_path = os.path.join(self.xml_dir, xml_file)
        boxes, labels = self.parse_xml(xml_path)

        boxes = self.resize_boxes(boxes, original_size, image.shape[1:])
        return image, {'boxes': boxes, 'labels': labels}

    def parse_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall('object'):
            label_name = obj.find('name').text.lower()

            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            boxes.append([xmin, ymin, xmax, ymax])

            if label_name in ['breast', 'meme']:
                labels.append(1)
            else:
                labels.append(0)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        return boxes, labels

    def resize_boxes(self, boxes, original_size, new_size):
        orig_h, orig_w = original_size
        new_h, new_w = new_size
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * (new_w / orig_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * (new_h / orig_h)

        return boxes

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc_class = nn.Linear(512, 2)
        self.fc_bbox = nn.Linear(512, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        class_output = self.fc_class(x)
        bbox_output = self.fc_bbox(x)
        return class_output, bbox_output


if __name__ == '__main__':
    net = Net().to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
    ])

    train_dataset = CustomDataset(train_files, png_dir, xml_dir, transform=transform)
    test_dataset = CustomDataset(test_files, png_dir, xml_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    criterion_bbox = nn.SmoothL1Loss()

    optimizer = optim.Adam(net.parameters(), lr=0.001)

    num_epochs = 120

    # Modeli kaydetme
    model_path = "trained_model.pth"
    torch.save(net.state_dict(), model_path)
    print(f"Model {model_path} yolunda kaydedildi.")

    # Kullanıcıya model eğitme veya kaydedilmiş ağırlıkları kullanma sorusu
    answer = input("Modeli tekrar eğitmek istiyor musunuz? (yes No): ").strip().lower()

    if answer == 'yes' or answer == 'y':
        # Modeli tekrar eğitme
        for epoch in range(num_epochs):
            print(f"Epochs: [{epoch + 1}/{num_epochs}]")
            for i, data in enumerate(train_loader, 0):
                inputs, targets = data
                inputs = inputs.to(device)
                boxes = targets['boxes'].to(device)

                optimizer.zero_grad()

                _, outputs_bbox = net(inputs)

                # Fazladan boyutu sıkıştır
                boxes = boxes.squeeze(1)

                loss_bbox = criterion_bbox(outputs_bbox, boxes)
                loss_bbox.backward()
                optimizer.step()

        torch.save(net.state_dict(), model_path)
        print(f"Model yeniden eğitildi ve {model_path} yolunda kaydedildi.")
    else:
        # Kaydedilmiş ağırlıkları yükleme
        try:
            net.load_state_dict(torch.load(model_path))
            net.eval()
            print(f"Model kaydedilmiş ağırlıkları {model_path} yolundan yükledi ve değerlendiriliyor.")
        except FileNotFoundError:
            print(f"Hata: {model_path} bulunamadı. Modeli tekrar eğitmeniz gerekebilir.")

    # Test adımı
    net.eval()

    total_class_loss = 0.0
    total_bbox_loss = 0.0

    print(f"outputs_bbox type: {type(outputs_bbox)}, shape: {getattr(outputs_bbox, 'shape', 'N/A')}")
    print(f"boxes type: {type(boxes)}, shape: {getattr(boxes, 'shape', 'N/A')}")

    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs, targets = data
            inputs = inputs.to(device)
            boxes = targets['boxes'].to(device)

            _, outputs_bbox = net(inputs)

            # Fazladan boyutu sıkıştır
            boxes = boxes.squeeze(1)

            loss_bbox = criterion_bbox(outputs_bbox, boxes)
            total_bbox_loss += loss_bbox.item()

            for j in range(10):
                if i == 0:
                    img = inputs[j].squeeze(0).cpu().numpy()
                    pred_box = torch.clamp(outputs_bbox[j], min=0, max=224).cpu().numpy()
                    true_box = boxes[j].cpu().numpy()

                    iou = calculate_iou(pred_box, true_box)

                    print(f"IoU Score: {iou}")

                    fig, ax = plt.subplots(1)
                    ax.imshow(img, cmap='gray')

                    xmin, ymin, xmax, ymax = pred_box
                    rect = patches.Rectangle(
                        (xmin, ymin), xmax - xmin, ymax - ymin,
                        linewidth=4, edgecolor='r', facecolor='none', label='Prediction'
                    )
                    ax.add_patch(rect)

                    xmin, ymin, xmax, ymax = true_box
                    rect = patches.Rectangle(
                        (xmin, ymin), xmax - xmin, ymax - ymin,
                        linewidth=2, edgecolor='g', facecolor='none', label='Ground Truth'
                    )
                    ax.add_patch(rect)

                    plt.legend()
                    plt.show()

    num_batches = len(test_loader)
    print(f"Average Bounding Box Loss: {total_bbox_loss / num_batches:.4f}")

import cv2
import os
import xml.etree.ElementTree as ET
from PIL import Image

def draw_bounding_boxes(image_dir, annotation_dir, output_dir):
    """
    PNG görüntülerine bounding box çizer ve çıktıyı kaydeder.
    :param image_dir: PNG görüntülerin olduğu klasör.
    :param annotation_dir: XML dosyalarının olduğu klasör.
    :param output_dir: Bounding box çizilmiş görüntülerin kaydedileceği klasör.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # PNG dosyalarını işle
    for image_file in os.listdir(image_dir):
        if not image_file.endswith(".png"):
            continue

        # Görüntü dosyasını oku
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)

        # XML dosyasını oku
        annotation_file = image_file.replace(".png", ".xml")
        annotation_path = os.path.join(annotation_dir, annotation_file)
        if not os.path.exists(annotation_path):
            print(f"XML dosyası bulunamadı: {annotation_path}")
            continue

        tree = ET.parse(annotation_path)
        root = tree.getroot()

        # Bounding box bilgilerini al
        for obj in root.findall("object"):
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)

            # Bounding box'u görüntü üzerine çiz
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # Yeşil renk

            # Etiket adı ekle
            label = obj.find("breast").text
            cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Çıktıyı kaydet
        output_path = os.path.join(output_dir, image_file)
        cv2.imwrite(output_path, image)

    print(f"Bounding box çizimleri {output_dir} klasörüne kaydedildi.")

# Kullanım
image_dir = r"C:\Users\canla\Desktop\donusum.png"  # PNG görüntülerin klasörü
annotation_dir = r"C:\Users\canla\Desktop\donusum.xml"  # XML dosyalarının klasörü
output_dir = r"C:\Users\canla\Desktop\bndngBox"  # Çizilmiş görüntülerin kaydedileceği klasör

draw_bounding_boxes(image_dir, annotation_dir, output_dir)

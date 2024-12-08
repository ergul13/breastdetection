import os
import shutil
import random


def split_data(input_dir, output_dir, train_ratio=0.6, test_ratio=0.2, val_ratio=0.2):

    # PNG ve XML dosyalarını listele
    png_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    xml_files = [f for f in os.listdir(input_dir) if f.endswith('.xml')]

    # PNG ve XML dosya adlarını kontrol et
    png_files_set = set([os.path.splitext(f)[0] for f in png_files])  # Sadece dosya adları
    xml_files_set = set([os.path.splitext(f)[0] for f in xml_files])

    # Eşleşen dosyalar (hem PNG hem XML mevcut olanlar)
    matched_files = png_files_set & xml_files_set

    # Eksik dosyaları yazdır
    missing_xml = png_files_set - xml_files_set
    missing_png = xml_files_set - png_files_set

    if missing_xml:
        print(f"UYARI: Aşağıdaki PNG dosyaları için XML dosyası eksik: {missing_xml}")
    if missing_png:
        print(f"UYARI: Aşağıdaki XML dosyaları için PNG dosyası eksik: {missing_png}")

    # Sadece eşleşen dosyaları kullan
    png_files = [f"{name}.png" for name in matched_files]
    xml_files = [f"{name}.xml" for name in matched_files]

    # Eşleşen dosya sayısını kontrol et
    if len(png_files) != len(xml_files):
        raise ValueError("Eşleşen PNG ve XML dosyalarının sayısı hâlâ farklı!")

    # PNG ve XML dosyalarını eşleştir ve karıştır
    paired_files = list(zip(png_files, xml_files))
    random.shuffle(paired_files)  # Dosyaları rastgele sırayla karıştır

    # Train, test ve validation setlerini böl
    total_files = len(paired_files)
    train_size = int(total_files * train_ratio)
    test_size = int(total_files * test_ratio)

    train_files = paired_files[:train_size]
    test_files = paired_files[train_size:train_size + test_size]
    val_files = paired_files[train_size + test_size:]

    # Çıkış klasörlerini oluştur
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    val_dir = os.path.join(output_dir, "validation")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Dosyaları yeni klasörlere taşı
    for folder, files in zip([train_dir, test_dir, val_dir], [train_files, test_files, val_files]):
        for png_file, xml_file in files:
            shutil.copy(os.path.join(input_dir, png_file), os.path.join(folder, png_file))  # PNG dosyasını kopyala
            shutil.copy(os.path.join(input_dir, xml_file), os.path.join(folder, xml_file))  # XML dosyasını kopyala

    # Sonuçları döndür
    return {
        "train_path": train_dir,
        "test_path": test_dir,
        "validation_path": val_dir
    }


# Örnek Kullanım
if __name__ == "__main__":
    input_dir = r"C:\Users\canla\Desktop\Yeni klasör"  # PNG ve XML dosyalarının bulunduğu klasör
    output_dir = r"C:\Users\canla\Desktop\ayrilmis_veriler"  # Ayrılmış verilerin kaydedileceği klasör

    paths = split_data(input_dir, output_dir)

    # Ayrılmış dosya yollarını yazdır
    print("Eğitim verileri yolu:", paths["train_path"])
    print("Test verileri yolu:", paths["test_path"])
    print("Validasyon verileri yolu:", paths["validation_path"])

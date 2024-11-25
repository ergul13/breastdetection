import pydicom
import numpy as np
from PIL import Image
import os
from pydicom.pixel_data_handlers.util import apply_windowing


def get_dicom_image_pixel_dimensions(dicom_path):
    ds = pydicom.dcmread(dicom_path)
    pixel_spacing = ds.PixelSpacing if 'PixelSpacing' in ds else None
    rows = ds.Rows
    columns = ds.Columns
    return pixel_spacing, rows, columns


def apply_windowing(data, dicom):
    try:
        window_center = dicom.WindowCenter
        window_width = dicom.WindowWidth
        intercept = dicom.RescaleIntercept
        slope = dicom.RescaleSlope
        if isinstance(window_center, pydicom.multival.MultiValue):
            window_center = window_center[0]
        if isinstance(window_width, pydicom.multival.MultiValue):
            window_width = window_width[0]
    except AttributeError:
        window_center = 40
        window_width = 80
        intercept = 0
        slope = 1

    img = (data * slope + intercept)
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)
    return img


def convert_dicom(dicom_path, output_dir):
    dicom = pydicom.dcmread(dicom_path)
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(dicom.pixel_array) - dicom.pixel_array
    else:
        data = dicom.pixel_array
    img = apply_windowing(data, dicom)
    img = (img.astype(float) - img.min()) / (img.max() - img.min())
    img = Image.fromarray((img * 255).astype(np.uint8))

    img_name, _ = os.path.splitext(os.path.basename(dicom_path))
    png_path = os.path.join(output_dir, f"{img_name}.png")
    os.makedirs(output_dir, exist_ok=True)
    img.save(png_path)
    return png_path


# Giriş ve çıkış klasör yollarını belirleyin
input_dir = r"C:\Users\canla\Desktop\Ergul"  # Giriş klasörü
output_dir = r"C:\Users\canla\Desktop\çıkış"  # Çıkış klasörü

# Giriş klasöründeki tüm DICOM dosyalarını al
for filename in os.listdir(input_dir):
    if filename.endswith(".dcm"):  # Sadece DICOM dosyalarını al
        dicom_path = os.path.join(input_dir, filename)

        # DICOM dosyasını PNG'ye dönüştür
        png_path = convert_dicom(dicom_path, output_dir)
        print(f"Converted {dicom_path} to {png_path}")

# -*- coding: utf-8 -*-
"""Запуск модели оценки глубины на новой картинке"""

import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# ==================== МОДЕЛЬ U-NET (та же архитектура) ====================
class UNetDepth(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.bottleneck = self.conv_block(512, 1024)
        
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec4 = self.conv_block(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = self.conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = self.conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = self.conv_block(128, 64)
        self.final = nn.Conv2d(64, 1, 1)
    
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(nn.MaxPool2d(2)(e1))
        e3 = self.enc3(nn.MaxPool2d(2)(e2))
        e4 = self.enc4(nn.MaxPool2d(2)(e3))
        b = self.bottleneck(nn.MaxPool2d(2)(e4))
        
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.final(d1)


# ==================== ФУНКЦИЯ ДЛЯ ПРЕДСКАЗАНИЯ ====================
def predict_depth(image_path, model_path='depth_estimation_model.pth', output_path=None):
    """
    Предсказание карты глубины для одного изображения
    
    Args:
        image_path: путь к входному изображению
        model_path: путь к файлу модели (.pth)
        output_path: путь для сохранения результата (опционально)
    
    Returns:
        depth_map: массив с предсказанной глубиной
    """
    # Проверка существования файлов
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Изображение не найдено: {image_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель не найдена: {model_path}")
    
    # Определяем устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    
    # Загрузка модели
    print("Загрузка модели...")
    model = UNetDepth().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Модель загружена")
    
    # Загрузка и подготовка изображения
    print(f"Загрузка изображения: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_size = img_rgb.shape[:2]
    
    # Изменяем размер до 256x256 (как при обучении)
    img_resized = cv2.resize(img_rgb, (256, 256))
    
    # Нормализация и преобразование в тензор
    img_tensor = torch.from_numpy(img_resized.astype(np.float32) / 255.0)
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Предсказание
    print("Вычисление карты глубины...")
    with torch.no_grad():
        depth_pred = model(img_tensor).squeeze().cpu().numpy()
    
    # Восстанавливаем оригинальный размер (опционально)
    depth_resized = cv2.resize(depth_pred, (original_size[1], original_size[0]))
    
    # Визуализация
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_rgb)
    axes[0].set_title('Исходное изображение')
    axes[0].axis('off')
    
    im1 = axes[1].imshow(depth_pred, cmap='plasma')
    axes[1].set_title('Карта глубины (256×256)')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    im2 = axes[2].imshow(depth_resized, cmap='plasma')
    axes[2].set_title(f'Карта глубины ({original_size[1]}×{original_size[0]})')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    
    plt.tight_layout()
    
    # Сохранение результата
    if output_path is None:
        output_path = 'depth_prediction_result.png'
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Результат сохранён: {output_path}")
    
    plt.show()
    
    return depth_resized


# ==================== ФУНКЦИЯ ДЛЯ БАТЧ-ОБРАБОТКИ ====================
def predict_batch(image_folder, model_path='depth_estimation_model.pth', output_folder='results'):
    """
    Обработка всех изображений в папке
    
    Args:
        image_folder: папка с изображениями
        model_path: путь к модели
        output_folder: папка для сохранения результатов
    """
    os.makedirs(output_folder, exist_ok=True)
    
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    images = [f for f in os.listdir(image_folder) if f.lower().endswith(supported_formats)]
    
    print(f"Найдено изображений: {len(images)}")
    
    for i, img_name in enumerate(images):
        print(f"\n[{i+1}/{len(images)}] Обработка: {img_name}")
        img_path = os.path.join(image_folder, img_name)
        output_path = os.path.join(output_folder, f'depth_{img_name}')
        
        try:
            predict_depth(img_path, model_path, output_path)
        except Exception as e:
            print(f"Ошибка при обработке {img_name}: {e}")


# ==================== КОМАНДНАЯ СТРОКА ====================
if __name__ == "__main__":
    print("=" * 50)
    print("Оценка глубины с помощью U-Net")
    print("=" * 50)
    
    if len(sys.argv) < 2:
        print("\nИспользование:")
        print("  Для одного изображения:")
        print("    python inference.py путь_к_изображению")
        print("\n  Для папки с изображениями:")
        print("    python inference.py --batch путь_к_папке")
        print("\nПримеры:")
        print("    python inference.py test.jpg")
        print("    python inference.py --batch ./my_images")
        sys.exit(0)
    
    # Параметры по умолчанию
    model_path = 'depth_estimation_model.pth'
    
    # Проверка аргументов
    if sys.argv[1] == '--batch':
        if len(sys.argv) < 3:
            print("Ошибка: укажите путь к папке с изображениями")
            sys.exit(1)
        predict_batch(sys.argv[2], model_path)
    else:
        image_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        predict_depth(image_path, model_path, output_path)

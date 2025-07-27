import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# Путь к изначальному датасету
source_dir = Path('sign_language_dataset')

# Папки, куда будем сохранять трейн и валидацию
train_dir = Path('sign_language_dataset_train')
val_dir = Path('sign_language_dataset_val')

# # Создаём папки, если их нет
# train_dir.mkdir(parents=True, exist_ok=True)
# val_dir.mkdir(parents=True, exist_ok=True)

# Для каждого класса (папка внутри sign_language_dataset)
for class_dir in source_dir.iterdir():
    if class_dir.is_dir():
        images = list(class_dir.glob('*.jpg'))

        # Делим на train и val
        train_imgs, val_imgs = train_test_split(images, test_size=0.3, random_state=42)

        # Куда класть
        train_class_dir = train_dir / class_dir.name
        val_class_dir = val_dir / class_dir.name
        train_class_dir.mkdir(parents=True, exist_ok=True)
        val_class_dir.mkdir(parents=True, exist_ok=True)

        # Копируем файлы
        for img_path in train_imgs:
            shutil.copy(img_path, train_class_dir / img_path.name)

        for img_path in val_imgs:
            shutil.copy(img_path, val_class_dir / img_path.name)

print("Разделение завершено: 70% train, 30% val.")

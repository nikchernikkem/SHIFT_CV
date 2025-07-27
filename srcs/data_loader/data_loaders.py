from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SignDataset(Dataset):
    def __init__(self, paths: List[Path], transform=None):
        self.paths = paths
        self.transform = transform # если есть аугментации

        # labels = sorted(set(str(x).split('/')[-2] for x in paths))

# На Windows в путях разделитель — обратный слэш \, а не /.
# str(x) на Windows даёт путь с \ (например: C:\Users\...), 
# поэтому split по '/' не сработает — список будет из одного элемента, и [-2] вызовет ошибку.

        labels = sorted(set(x.parent.name for x in paths))
        self.one_hot_encoding = {label: i for i, label in enumerate(labels)}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = cv2.imread(str(self.paths[idx]))
        # label = str(self.paths[idx]).split('/')[-2]
        label = self.paths[idx].parent.name
        image = cv2.resize(image, (300, 300)) # 200 на 200 для мобилки
        image = np.transpose(image, (2, 0, 1))

        return torch.tensor(image).float(), torch.tensor(self.one_hot_encoding[label])


# def get_sign_dataloader(
#         path_train, path_val, batch_size, shuffle=True, num_workers=1,
#     ):
#     train_dataset = SignDataset(paths=[*Path(path_train).rglob('*.jpg')])
#     val_dataset = SignDataset(paths=[*Path(path_val).rglob('*.jpg')])

#     loader_args = {
#         'batch_size': batch_size,
#         'shuffle': shuffle,
#         'num_workers': num_workers
#     }
#     return DataLoader(train_dataset, **loader_args), DataLoader(val_dataset, **loader_args)

from pathlib import Path

def get_sign_dataloader(path_train, path_val, batch_size, shuffle=True, num_workers=1):
    train_paths = list(Path(path_train).rglob('*.jpg'))
    val_paths = list(Path(path_val).rglob('*.jpg'))
    print(f"Train images found: {len(train_paths)}")
    print(f"Val images found: {len(val_paths)}")

    train_dataset = SignDataset(paths=train_paths)
    val_dataset = SignDataset(paths=val_paths)

    loader_args = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers
    }
    return DataLoader(train_dataset, **loader_args), DataLoader(val_dataset, **loader_args)


def get_sign_test_dataloader(
        path_test, batch_size, num_workers=1,
    ):
    test_dataset = SignDataset(paths=[*Path(path_test).rglob('*.jpg')])

    loader_args = {
        'batch_size': batch_size,
        'shuffle': False,
        'num_workers': num_workers
    }
    return DataLoader(test_dataset, **loader_args)

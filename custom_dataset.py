import torch
from torch.utils.data import Dataset
import os

class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))

    def __len__(self):
        length = 0
        for class_dir in self.classes:
            class_path = os.path.join(self.root_dir, class_dir)
            length += len(os.listdir(class_path))
        return length

    def __getitem__(self, idx):
        current_idx = 0
        for class_dir in self.classes:
            class_path = os.path.join(self.root_dir, class_dir)
            files = os.listdir(class_path)
            num_files = len(files)
            if idx < current_idx + num_files:
                file_name = files[idx - current_idx]
                file_path = os.path.join(class_path, file_name)
                tensor_data = torch.load(file_path)
                return tensor_data, self.classes.index(class_dir)  # Assuming class directory name is class label
            else:
                current_idx += num_files

        raise IndexError("Index out of bounds")

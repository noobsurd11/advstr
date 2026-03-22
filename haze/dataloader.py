import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset

class PairedOHazeDataset(Dataset):
    def __init__(self, root_dir, split='train', size=512):
        self.clear_dir = os.path.join(root_dir, split, 'clear')
        self.hazy_dir  = os.path.join(root_dir, split, 'hazy')

        self.files = sorted(os.listdir(self.clear_dir))
        assert len(self.files) > 0, f"No images found in {self.clear_dir}"

        self.size = size
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        clear_fn = self.files[idx]              # e.g. "26_outdoor_GT.jpg"
        
        # Build hazy filename
        # Replace "_GT" with "_hazy"
        hazy_fn = clear_fn.replace("_GT", "_hazy")

        clear_path = os.path.join(self.clear_dir, clear_fn)
        hazy_path  = os.path.join(self.hazy_dir,  hazy_fn)

        # Safety check
        if not os.path.exists(hazy_path):
            raise FileNotFoundError(f"Hazy file not found for: {clear_fn} → expected {hazy_fn}")

        J = Image.open(clear_path).convert('RGB')
        I = Image.open(hazy_path).convert('RGB')

        J = self.transform(J)
        I = self.transform(I)

        return J, I, clear_fn

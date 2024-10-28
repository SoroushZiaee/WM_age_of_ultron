import os
from torch.utils.data import Dataset
from torchvision import transforms as T
import PIL.Image


class MuriDataset(Dataset):
    def __init__(self, root: str, transforms=None, return_paths=False):

        self.root = root

        self.transforms = transforms
        img_path_list = os.listdir(root)
        img_path_list.sort()

        self.img_path_list = img_path_list
        self.return_path = return_paths

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root, self.img_path_list[idx])
        # print(f"{img_name = }")
        image = PIL.Image.open(img_name).convert("RGB")

        if self.transforms:
            image = self.transforms(image)

        if self.return_path:
            return image, os.path.basename(img_name)

        return image

from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from .preprocess import preprocess
from .config import CFG
import os


class SkinDataset(Dataset):
    def __init__(self, root_dir, train=True):
        """
        Args:
            root_dir (Path): Path to 'train' or 'val' folder.
            train (bool): If True, applies heavy augmentations
        """
        self.root_dir = root_dir
        self.items = []
        self.train = train

        # 1. Scan the folders
        print(f"📂 Scanning dataset at: {self.root_dir}")

        for class_name, label_idx in CFG.LABEL_MAP.items():
            class_path = self.root_dir / class_name

            if not class_path.exists():
                print(f"⚠️ Warning: Folder not found: {class_path}")
                continue

            files = os.listdir(class_path)
            count = 0
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.items.append((class_path / f, label_idx))
                    count += 1
            print(f"   found {count} images in '{class_name}'")

        if len(self.items) == 0:
            raise RuntimeError(f"❌ No images found in {self.root_dir}. Check your path!")

        # 2. ENHANCED AUGMENTATIONS
        if train:
            self.transform = A.Compose([
                A.Resize(CFG.IMAGE_SIZE, CFG.IMAGE_SIZE),

                # Geometric augmentations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2,
                                   rotate_limit=45, p=0.5),
                A.Transpose(p=0.3),

                # Color augmentations (crucial for medical images)
                A.OneOf([
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30,
                                         val_shift_limit=20, p=1.0),
                    A.RandomBrightnessContrast(brightness_limit=0.2,
                                               contrast_limit=0.2, p=1.0),
                    A.CLAHE(clip_limit=4.0, p=1.0),
                ], p=0.5),

                # Advanced augmentations
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                ], p=0.3),

                # Elastic transformations (simulate skin deformation)
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),

                # Coarse dropout (simulate occlusions)
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32,
                                min_holes=1, fill_value=0, p=0.3),

                # Normalization
                A.Normalize(mean=CFG.MEAN, std=CFG.STD),
                ToTensorV2()
            ])
        else:
            # Validation/Test: minimal augmentation
            self.transform = A.Compose([
                A.Resize(CFG.IMAGE_SIZE, CFG.IMAGE_SIZE),
                A.Normalize(mean=CFG.MEAN, std=CFG.STD),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, label = self.items[idx]

        try:
            img = np.array(Image.open(img_path).convert("RGB"))
        except Exception as e:
            print(f"❌ Corrupt image: {img_path}")
            # Fallback to prevent crash
            return self.__getitem__((idx + 1) % len(self))

        # Apply preprocessing
        img = preprocess(img)

        # Apply augmentations
        img = self.transform(image=img)["image"]

        return img, label
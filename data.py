from pathlib import Path

from torch.utils.data import Dataset
import torchvision.transforms as T
from functools import partial
from PIL import Image
from torch import nn

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

class ImageDataset(Dataset):
    def __init__(
        self,
        folder: str | Path,
        image_size: int,
        exts: list[str] = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        if isinstance(folder, str):
            folder = Path(folder)

        assert folder.is_dir()

        self.folder = folder
        self.image_size = image_size

        self.paths = [p for ext in exts for p in folder.glob(f'**/*.{ext}')]

        def convert_image_to_fn(img_type, image):
            if image.mode == img_type:
                return image

            return image.convert(img_type)

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

class PairedImageDataset(Dataset):
    def __init__(
        self,
        folder: str | Path,
        image_size: int,
        exts: list[str] = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip=False,
        convert_image_to=None
    ):
        super().__init__()

        if isinstance(folder, str):
            folder = Path(folder)

        assert folder.is_dir()

        # 输入和GT图像目录
        self.input_folder = folder / 'input'
        self.gt_folder = folder / 'target'

        assert self.input_folder.is_dir() and self.gt_folder.is_dir()

        self.image_size = image_size

        # 获取所有GT图像的路径
        self.gt_paths = [p for ext in exts for p in self.gt_folder.glob(f'**/*.{ext}')]

        # 创建一个字典，将GT图像路径与输入图像路径对应起来
        self.pairs = []
        for gt_path in self.gt_paths:
            # 查找与gt图像同名的input图像
            input_path = self.input_folder / gt_path.name
            if input_path.exists():
                self.pairs.append((input_path, gt_path))

        # 转换函数
        def convert_image_to_fn(img_type, image):
            if image.mode == img_type:
                return image
            return image.convert(img_type)

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if convert_image_to else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        # 获取图像对
        input_path, gt_path = self.pairs[index]

        # 打开输入和目标图像
        input_img = Image.open(input_path)
        gt_img = Image.open(gt_path)

        # 应用变换
        input_img = self.transform(input_img)
        gt_img = self.transform(gt_img)

        return input_img, gt_img

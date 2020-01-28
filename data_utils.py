import torch.utils.data as data

from os import listdir
from os.path import join
from PIL import Image, ImageFilter


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_target_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    img = img.filter(ImageFilter.SHARPEN);
    img.save("target.jpg")
    return img

def load_train_img(filepath):
    img = Image.open(filepath)
    img = img.filter(ImageFilter.GaussianBlur(1))
    img.convert('YCbCr')
    img.save("sample.jpg")
    return img

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        file_path = self.image_filenames[index]
        if self.input_transform:
            img = Image.open(file_path)
            input = img.filter(ImageFilter.GaussianBlur(1))
            input.convert('YCbCr')
            input.save("sample.jpg")
            input = self.input_transform(input)

        if self.target_transform:
            img = Image.open(file_path).convert('YCbCr')
            target = img.filter(ImageFilter.SHARPEN);
            target.save("target.jpg")
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)


import torch.nn
import imageio
import os
import cv2
import torch.utils.data
import torchvision


class PanSharpeningDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform

        self.images_pan = list(sorted(os.listdir(os.path.join(self.root, 'pan'))))
        self.images_pan.sort(key=lambda x: int(x[:-4]))
        self.images_ms = list(sorted(os.listdir(os.path.join(self.root, 'ms'))))
        self.images_ms.sort(key=lambda x: int(x[:-4]))
        self.images_ms_label = list(sorted(os.listdir(os.path.join(self.root, 'ms_label'))))
        self.images_ms_label.sort(key=lambda x: int(x[:-4]))
        self.images_pan_label = list(sorted(os.listdir(os.path.join(self.root, 'pan_label'))))
        self.images_pan_label.sort(key=lambda x: int(x[:-4]))

    def __getitem__(self, item):
        image_pan_path = os.path.join(self.root, 'pan', self.images_pan[item])
        image_ms_path = os.path.join(self.root, 'ms', self.images_ms[item])
        image_ms_label_path = os.path.join(self.root, 'ms_label', self.images_ms_label[item])
        image_pan_label_path = os.path.join(self.root, 'pan_label', self.images_pan_label[item])

        image_pan = imageio.imread(image_pan_path) 
        image_ms = imageio.imread(image_ms_path)  # B G R NIR
        image_ms_label = imageio.imread(image_ms_label_path)
        image_pan_label = imageio.imread(image_pan_label_path)

        if self.transform is not None:
            image_pan = self.transform(image_pan)
            image_ms = self.transform(image_ms)
            image_pan_label = self.transform(image_pan_label)
            image_ms_label = self.transform(image_ms_label)
            pass

        return image_pan, image_ms, image_pan_label, image_ms_label

    def __len__(self):
        return len(self.images_ms_label)


if __name__ == '__main__':
    print("Hello world")

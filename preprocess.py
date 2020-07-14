import torch
import torchvision

from PIL import Image

import csv
import os

class get_image_folder(torch.utils.data.Dataset):
    def __init__(self, data_path, subdir, img_dim, train=True):
        self.total_path = '/'.join((data_path, subdir))
        self.label = []
        self.img_dim = img_dim
        self.train = train

        if train == False:
            path = os.path.join(data_path, subdir)
            for file_path in os.listdir(path):
                self.label.append(os.path.join(path, file_path))
            return

        # Get labels
        with open('/'.join((self.total_path, 'labels.csv'))) as csv_file:
            reader = csv.reader(csv_file, delimiter=',')

            for datum in reader:
                self.label.append(datum)
        
    def __getitem__(self, index):        
        # Error: Should not happen
        if self.train and len(self.label[index]) != 2:
            raise RuntimeError('The ' + str(i) + '\'th entry of labels.csv does not have exactly 2 columns')

        raw_data = []
        if self.train:
            raw_data = Image.open(self.label[index][0]).convert(mode='RGB')
        else:
            raw_data = Image.open(self.label[index]).convert(mode='RGB')

        resized_img = raw_data.resize(self.img_dim)

        print(self.label[index])
        print(resized_img.size)

        img = torchvision.transforms.ToTensor().__call__(resized_img)

        if self.train:
            count = float(self.label[index][1]) # convert count to float
            return (img, count)
        else:
            return img

    def __len__(self):
        return len(self.label)

def get_loader(data_path, subdir, img_dim, batch_size, num_workers=4, shuffle=True, train=True):
    dataset = get_image_folder(data_path, subdir, img_dim, train)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         shuffle=shuffle)
    return loader


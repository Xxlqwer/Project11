#-*- codeing = utf-8 -*-
#@Time:2023/12/14 13:47
#@Author:LXX
#@File: note.py
#@Software:

import csv
import os
'''
# 图片种类和文件夹路径
image_categories = ["0", "1", "2", "3", "4"]

# folder_paths = ["D:/023/machine/123/train/Arborio", "D:/023/machine/123/train/Basmati", "D:/023/machine/123/train/Ipsala", "D:/023/machine/123/train/Jasmine", "D:/023/machine/123/train/Karacadag"]
folder_paths = ["D:/023/machine/123/test/Arborio", "D:/023/machine/123/test/Basmati", "D:/023/machine/123/test/Ipsala", "D:/023/machine/123/test/Jasmine", "D:/023/machine/123/test/Karacadag"]
# test_path = r"D:\023\machine\123\test_data.csv"
# print(test_path)
# 创建或打开一个 CSV 文件
csv_file = open('test_data.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)

# 写入表头
csv_writer.writerow(['0', '1'])

# 遍历文件夹，将文件路径写入 CSV 文件
for index, folder_path in enumerate(folder_paths):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        csv_writer.writerow([f'{file_path}', f'{image_categories[index]}'])

# 关闭 CSV 文件
csv_file.close()

'''
'''
import os
 # 4500-3642=858
folders1 = ['D:/023/machine/123/test/Arborio', 'D:/023/machine/123/test/Basmati', 'D:/023/machine/123/test/Ipsala', 'D:/023/machine/123/test/Jasmine', 'D:/023/machine/123/test/Karacadag']  # 替换成你的文件夹路径列表
  # 10500-8500=2000
folders2 = ['D:/023/machine/123/train/Arborio', 'D:/023/machine/123/train/Basmati', 'D:/023/machine/123/train/Ipsala', 'D:/023/machine/123/train/Jasmine', 'D:/023/machine/123/train/Karacadag']  # 替换成你的文件夹路径列表

photos_to_delete = 3642  # 每个文件夹中要删除的照片数量

for folder in folders1:
    if not os.path.exists(folder):
        print(f"{folder} 不存在")
        continue

    files = os.listdir(folder)
    photo_count = 0

    for file_name in files:
        if photo_count < photos_to_delete and file_name.endswith('.jpg'):  # 按你的需求修改文件类型
            os.remove(os.path.join(folder, file_name))
            photo_count += 1

    print(f"{folders1} 中的 {photo_count} 张照片被删除")
'''
# import datetime
# print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn as nn

import pandas as pd
from PIL import Image
import time

import model
import csv
import os

import datetime



class dataset(Dataset):
    def __init__(self, path):
        self.path = path
        csv_list = pd.read_csv(self.path)
        self.x, self.y = csv_list['0'], csv_list['1']

        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    def __len__(self):
        # 返回数据集的大小
        return len(self.x)  # 假设 self.x 是你数据集的一个属性，表示样本的数量

    def __getitem__(self, index):
        X = Image.open(self.x[index]).convert('RGB')
        Y = self.y[index]
        label_y = torch.zeros(5)
        label_y[Y] = 1
        X = self.transform(X)

        return X, label_y


def get_data_iter(train_path, test_path):
    train_dataset = dataset(train_path)
    test_dataset = dataset(test_path)
    train_iter = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_iter = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return train_iter, test_iter


def train(model, optimizer, loss, train_iter, test_iter):
    for epoch in range(10000):
        train_loss = 0
        test_loss = 0
        train_acc = 0
        test_acc = 0
        total_batch = 0
        test_batch=0
        start = time.time()
        for x, y in train_iter:
            x = x.cuda()
            y = y.cuda()
            model.train()
            out = model(x)

            optimizer.zero_grad()
            l = loss(out, y)
            l.backward()
            optimizer.step()

            total_batch += out.shape[0]
            train_acc += torch.sum(out.argmax(dim=1) == y.argmax(dim=1)).cpu()
            train_loss += l.cpu()
            # 进度 ,損失率，準確率，耗費時間
        if (epoch % 5 == 0):
            print(
            f"第{epoch}轮：train损失：{round(float(train_loss) / total_batch, 4)}%，"
            f"train准确率：{round(100 * int(train_acc) / total_batch, 2)}%，"
            f"耗费时间：{round(time.time() - start, 1)}s")


            with torch.no_grad():
                model.eval()
                for x, y in test_iter:
                    x = x.cuda()
                    y = y.cpu()
                    out = model(x).cpu()
                    test_batch += out.shape[0]  #
                    test_acc += sum(out.argmax(dim=1) == y.argmax(dim=1))
                    test_loss += loss(out, y).cpu()
                train_loss = train_loss / len(train_iter)
                test_loss = test_loss / len(test_iter)
                train_acc = 100 * train_acc / len(train_iter)
                test_acc = 100 * train_acc / len(test_iter)
                torch.save(model, f"D:/023/machine/123/result{epoch}.pth")
                print(
                    f"第{epoch}轮："
                    f"test损失：{round(float(test_loss) / test_batch, 4)}%，"
                    f"test准确率：{round(100 * int(test_acc) / test_batch, 2)}%，"
                    f"耗费时间：{round(time.time() - start, 1)}s，当前时间：{t2}")

        # if (epoch % 5 == 0):
        #     with torch.no_grad():
        #         model.eval()
        #         for x, y in test_iter:
        #             x = x.cuda()
        #             y = y.cpu()
        #             out = model(x).cpu()
        #             test_batch+=out
        #             test_acc += sum(out.argmax(dim=1) == y.argmax(dim=1))
        #             test_loss += loss(out, y).cpu()
        #         train_loss = train_loss / len(train_iter)
        #         test_loss = test_loss / len(test_iter)
        #         train_acc = 100 * train_acc / len(train_iter)
        #         test_acc = 100 * train_acc / len(test_iter)
        #     torch.save(model, f"D:/023/machine/123/result{epoch}.pth")





if __name__ == '__main__':
    train_path= "D:/023/machine/123/train_data.csv"
    test_path = "D:/023/machine/123/test_data.csv"
    model = model.ResNet().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_iter, test_iter = get_data_iter(train_path, test_path)
    loss = nn.CrossEntropyLoss().cuda()
    train(model, optimizer, loss, train_iter, test_iter)



# tensorboard, tqdm, attention, log, .pt



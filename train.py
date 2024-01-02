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
        return len(self.x)  # self.x 是你数据集的一个属性，表示样本的数量

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
    for epoch in range(1,10000):
        train_loss = 0
        test_loss = 0
        train_acc = 0
        test_acc = 0
        total_batch = 0
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

        with torch.no_grad():
            model.eval()
            for x, y in test_iter:
                x = x.cuda()
                y = y.cpu()
                out = model(x).cpu()
                l = loss(out, y)

                test_acc += torch.sum(out.argmax(dim=1) == y.argmax(dim=1)) # 121212
                test_loss += l

            train_loss /= len(train_iter.dataset)  # 使用整个数据集的大小
            train_accuracy = 100 * train_acc / len(train_iter.dataset)

            # 计算测试损失和准确率
            test_loss /= len(test_iter.dataset)
            test_accuracy = 100 * test_acc / len(test_iter.dataset)

            # 打印训练和测试的损失与准确率
            print(
                f"Epoch {epoch}: "
                f"train_loss: {train_loss:.4f}, train_accuracy: {train_accuracy:.2f}%, "
                f"test_loss: {test_loss:.4f}, test_accuracy: {test_accuracy:.2f}%"
                f"耗费时间：{round(time.time() - start, 1)}s，"
            )
        if(epoch%5==0):
            torch.save(model, f"D:/023/machine/123/result{epoch}.pth")


            # 进度 ,損失率，準確率，耗費時間
        # if (epoch % 5 == 0):
        #     # print(
        #     # f"第{epoch}轮：train损失：{round(float(train_loss) / total_batch, 4)}%，"
        #     # f"train准确率：{round(100 * int(train_acc) / total_batch, 2)}%，"
        #     # f"耗费时间：{round(time.time() - start, 1)}s，当前时间：{t1}\n")
        #     with torch.no_grad():
        #         model.eval()
        #         for x, y in test_iter:
        #             x = x.cuda()
        #             y = y.cpu()
        #             out = model(x).cpu()
        #             l = loss(out, y)
        #
        #             test_batch += out.shape[0]  # 121212
        #             # test_acc += torch.sum(out.argmax(dim=1) == y.argmax(dim=1))
        #             test_acc += torch.sum(out.argmax(dim=1) == y.argmax(dim=1)) # 121212
        #             test_loss += l
        #         #     t2 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')       # 121212
        #         # train_loss = train_loss / len(train_iter) # 121212
        #         # test_loss = test_loss / len(test_iter) # 121212
        #         # train_acc = 100 * train_acc / len(train_iter) # 121212
        #         # test_acc = 100 * train_acc / len(test_iter) # 121212
        #
        #     # torch.save(model, f"D:/023/machine/123/result{epoch}.pth")
        #     # 第0轮：train损失：0.0002 %，train准确率：6.2 %，
        #     # test损失：0.0014 %，test准确率：10.7 %，耗费时间：286.4
        #     # s，当前时间：2023 - 12 - 15
        #     # 21: 44:40
        #     print(
        #         f"第{epoch}轮："
        #         f"train损失：{round(float(train_loss) / total_batch, 4)}%，"    # 121212
        #         f"train准确率：{round(100 * int(train_acc) / total_batch, 2)}%，\n"        # 121212
        #         f"test损失：{round(float(test_loss) / test_batch, 4)}%，"        # 121212
        #         f"test准确率：{round(100 * int(test_acc) / test_batch, 2)}%，"        # 121212
        #         f"耗费时间：{round(time.time() - start, 1)}s，")



if __name__ == '__main__':
    train_path= "D:/023/machine/123/train_data.csv"
    test_path = "D:/023/machine/123/test_data.csv"
    model = model.ResNet().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_iter, test_iter = get_data_iter(train_path, test_path)
    loss = nn.CrossEntropyLoss().cuda()
    train(model, optimizer, loss, train_iter, test_iter)



# tensorboard, tqdm, log

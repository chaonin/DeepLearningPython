# cnn.py
# 零下载版：不依赖 MNIST 文件，直接用随机数据演示 CNN 训练流程

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ====== 1. 配置 ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 假装我们在做 MNIST：1 通道，28x28 图片，10 类
NUM_CLASSES = 10
IMAGE_SIZE = 28
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.01
TRAIN_SAMPLES = 5000   # 训练数据量（全是随机的）
TEST_SAMPLES = 1000    # 测试数据量（全是随机的）


# ====== 2. 随机数据集 ======
class RandomImageDataset(Dataset):
    """
    生成随机图片和标签，形状和 MNIST 一样：
    - 图片：1 x 28 x 28
    - 标签：0 ~ 9
    """

    def __init__(self, num_samples):
        super().__init__()
        self.num_samples = num_samples
        # 随机生成 [0,1) 的浮点数作为像素
        self.images = torch.rand(num_samples, 1, IMAGE_SIZE, IMAGE_SIZE)
        # 随机生成 0~9 的整数作为标签
        self.labels = torch.randint(0, NUM_CLASSES, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# ====== 3. 简单 CNN 模型 ======
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # 输入: 1 x 28 x 28
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # 16 x 28 x 28
            nn.ReLU(),
            nn.MaxPool2d(2),                             # 16 x 14 x 14

            nn.Conv2d(16, 32, kernel_size=3, padding=1), # 32 x 14 x 14
            nn.ReLU(),
            nn.MaxPool2d(2)                              # 32 x 7 x 7
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.classifier(x)
        return x


# ====== 4. 训练和评估 ======
def train_one_epoch(model, dataloader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        # 前向
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向 + 更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录指标
        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

        if (batch_idx + 1) % 50 == 0:
            print(
                f"Epoch [{epoch}] Step [{batch_idx+1}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f}"
            )

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    print(f"Epoch [{epoch}] Train Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    print(f"Test Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")
    return avg_loss, avg_acc


# ====== 5. 主函数 ======
def main():
    # 构造随机数据集
    train_dataset = RandomImageDataset(TRAIN_SAMPLES)
    test_dataset = RandomImageDataset(TEST_SAMPLES)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 模型、损失、优化器
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    print("Start training on random data ...")
    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        print("Evaluating on random test data ...")
        evaluate(model, test_loader, criterion)
        print("-" * 50)


if __name__ == "__main__":
    main()


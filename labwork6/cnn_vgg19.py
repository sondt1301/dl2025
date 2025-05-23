import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5
NUM_CLASSES = 10

vgg19_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']


class VGG19(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(VGG19, self).__init__()
        self.features = self._make_layers(vgg19_config)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for v_or_m in cfg:
            if v_or_m == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                num_out_channels = v_or_m
                conv2d = nn.Conv2d(in_channels, num_out_channels, kernel_size=3, padding=1)
                layers.append(conv2d)
                layers.append(nn.ReLU(inplace=True))
                in_channels = num_out_channels
        return nn.Sequential(*layers)

if __name__ == '__main__':
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    model = VGG19(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
        model.train()
        running_loss = 0.0
        correct_train_predictions = 0
        total_train_samples = 0

        for i, data_batch in enumerate(trainloader, 0):
            input_images, true_labels = data_batch
            input_images, true_labels = input_images.to(DEVICE), true_labels.to(DEVICE)

            optimizer.zero_grad()
            model_outputs = model(input_images)
            loss = criterion(model_outputs, true_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted_classes = torch.max(model_outputs.data, 1)
            total_train_samples += true_labels.size(0)
            correct_train_predictions += (predicted_classes == true_labels).sum().item()

            if (i + 1) % 100 == 0:
                avg_loss = running_loss / 100
                current_train_acc = 100 * correct_train_predictions / total_train_samples
                print(
                    f'  Batch [{i + 1}/{len(trainloader)}], Avg. Loss for last 100 batches: {avg_loss:.3f}, Training Acc so far: {current_train_acc:.2f}%')
                running_loss = 0.0

        epoch_train_accuracy = 100 * correct_train_predictions / total_train_samples
        print(f"--- Epoch {epoch + 1} Finished. Training Accuracy: {epoch_train_accuracy:.2f}% ---")

    model.eval()

    all_true_labels_list = []
    all_predicted_labels_list = []

    with torch.no_grad():
        for data_batch in testloader:
            test_images, test_labels = data_batch
            test_images, test_labels = test_images.to(DEVICE), test_labels.to(DEVICE)
            outputs_on_test_data = model(test_images)
            _, predicted_test_classes = torch.max(outputs_on_test_data.data, 1)
            all_true_labels_list.extend(test_labels.cpu().numpy())
            all_predicted_labels_list.extend(predicted_test_classes.cpu().numpy())

    accuracy = accuracy_score(all_true_labels_list, all_predicted_labels_list)
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        all_true_labels_list, all_predicted_labels_list, average='weighted', zero_division=0
    )

    print(f'\n--- Evaluation Results ---')
    print(f'Overall Accuracy on the test set: {accuracy * 100:.2f} %')
    print(f'Overall Weighted Precision: {precision_w:.4f}')
    print(f'Overall Weighted Recall:    {recall_w:.4f}')
    print(f'Overall Weighted F1-score:  {f1_w:.4f}')

    precision_pc, recall_pc, f1_pc, support_pc = precision_recall_fscore_support(
        all_true_labels_list, all_predicted_labels_list, average=None, labels=list(range(NUM_CLASSES)), zero_division=0
    )

    for i in range(NUM_CLASSES):
        print(f"\nClass: {classes[i]} (label {i})")
        print(f"  Precision: {precision_pc[i]:.4f}")
        print(f"  Recall:    {recall_pc[i]:.4f}")
        print(f"  F1-score:  {f1_pc[i]:.4f}")
        print(f"  Support:   {support_pc[i]}")
        print("-" * 30)

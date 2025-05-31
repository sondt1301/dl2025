import random
import time

# This is only used for data loading, metrics, and plotting
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt

from CNN.Conv2D import Conv2D
from CNN.ReLu import ReLU
from CNN.MaxPool2D import MaxPool2D
from CNN.Flatten import Flatten
from CNN.Dense import Dense
from CNN.SoftmaxCrossEntropy import SoftmaxCrossEntropy

# Read model config
def read_config(filepath="model_config.txt"):
    # Default config
    config = {
        'num_conv_layers': 1,
        'num_dense_layers': 1,
        'kernel_size': 1
    }
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            config[key] = int(value)
    return config

# Load mnist data
def load_data(target_classes, num_train_per_class, num_test_per_class):
    print(f"Loading MNIST subset including classes {target_classes}")
    transform_to_tensor = transforms.Compose([transforms.ToTensor()])
    try:
        full_trainset = torchvision.datasets.MNIST(root='./data_mnist', train=True, download=True, transform=transform_to_tensor)
        full_testset = torchvision.datasets.MNIST(root='./data_mnist', train=False, download=True, transform=transform_to_tensor)
    except Exception as e:
        print(f"Error loading data: {e}.")
        return None, None

    def extract_and_convert(dataset, req_classes, num_per_class):
        img_lists = []
        lbl_lists = []
        counts = {l: 0 for l in req_classes}
        lbl_map = {ol: nl for nl, ol in enumerate(req_classes)}

        shuffled_indices = list(range(len(dataset)))
        random.shuffle(shuffled_indices)

        for idx_shuffled in shuffled_indices:
            img_tensor, orig_lbl = dataset[idx_shuffled]
            if orig_lbl in req_classes and counts[orig_lbl] < num_per_class:
                img_lists.append([[r.tolist() for r in img_tensor[0]]])
                lbl_lists.append(lbl_map[orig_lbl])
                counts[orig_lbl] += 1
            if all(counts[l] >= num_per_class for l in req_classes):
                break

        combined = list(zip(img_lists, lbl_lists))
        random.shuffle(combined)
        return combined

    train_data = extract_and_convert(full_trainset, target_classes, num_train_per_class)
    test_data = extract_and_convert(full_testset, target_classes, num_test_per_class)
    print(f"Number of data in train set: {len(train_data)}, Number of data in test set: {len(test_data)}")
    return train_data, test_data


# Train and Test model
def run_model():
    model_config = read_config()

    img_depth, img_height, img_width = 1, 28, 28
    target_mnist_classes = (0, 1, 2)
    num_classes = len(target_mnist_classes)

    print(f"Number of convolutional layers ={model_config['num_conv_layers']}, Number of dens layers={model_config['num_dense_layers']}, Number of kernels={model_config['kernel_size']}")

    kernel_size = model_config['kernel_size']

    # Build model
    # Conv Layer 1
    conv1 = Conv2D(filter_size=kernel_size, input_depth=img_depth, stride=1, padding=1)
    relu_conv1 = ReLU()
    h_after_c1 = (img_height + 2 * conv1.padding - conv1.filter_size) // conv1.stride + 1
    w_after_c1 = (img_width + 2 * conv1.padding - conv1.filter_size) // conv1.stride + 1

    # Conv Layer 2
    conv2, relu_conv2 = None, None
    h_after_convs, w_after_convs = h_after_c1, w_after_c1
    if model_config['num_conv_layers'] == 2:
        conv2 = Conv2D(filter_size=kernel_size, input_depth=1, stride=1, padding=1)
        relu_conv2 = ReLU()
        h_after_convs = (h_after_c1 + 2 * conv2.padding - conv2.filter_size) // conv2.stride + 1
        w_after_convs = (w_after_c1 + 2 * conv2.padding - conv2.filter_size) // conv2.stride + 1

    pool1 = MaxPool2D(size=2, stride=2)
    pooled_h = (h_after_convs - pool1.size) // pool1.stride + 1
    pooled_w = (w_after_convs - pool1.size) // pool1.stride + 1

    flatten = Flatten()
    flattened_size = 1 * pooled_h * pooled_w

    # Dense Layers
    dense1_hidden_units = 32
    dense1, relu_dense1 = None, None

    if model_config['num_dense_layers'] == 1:
        dense_output = Dense(input_size=flattened_size, output_size=num_classes)
    elif model_config['num_dense_layers'] == 2:
        dense1 = Dense(input_size=flattened_size, output_size=dense1_hidden_units)
        relu_dense1 = ReLU()
        dense_output = Dense(input_size=dense1_hidden_units, output_size=num_classes)

    loss_fn = SoftmaxCrossEntropy()

    # Prepare data
    train_samples_per_class = 25
    test_samples_per_class = 10
    training_dataset, test_dataset = load_data(target_mnist_classes, train_samples_per_class, test_samples_per_class)

    lr = 0.001
    epochs = 50

    history = {'train_loss': [], 'train_acc': []}
    print(f"Start training ({len(training_dataset)} samples, {epochs} epochs)")

    # === TRAINING PHASE ===
    for epoch in range(epochs):
        epoch_start_time = time.time()
        total_loss_train, correct_predictions_train = 0.0, 0
        current_epoch_shuffled_data = list(training_dataset)
        random.shuffle(current_epoch_shuffled_data)

        for sample_idx, (img_volume, label_remapped) in enumerate(current_epoch_shuffled_data):
            # --- Forward Pass ---
            x = conv1.forward(img_volume)
            x = relu_conv1.forward(x)
            if conv2:
                x_c2_in = [x]
                x = conv2.forward(x_c2_in)
                x = relu_conv2.forward(x)
            x = pool1.forward(x)
            x_f_in = [x]
            x_flat = flatten.forward(x_f_in)
            if dense1:
                x_d1 = dense1.forward(x_flat)
                x_d1_2d = [x_d1]
                x_r_d1_2d = relu_dense1.forward(x_d1_2d)
                x = x_r_d1_2d[0]
            else:
                # If 1 layer, pass directly
                x = x_flat
            logits = dense_output.forward(x)

            loss = loss_fn.forward(logits, label_remapped)
            total_loss_train += loss
            pred_index = logits.index(max(logits)) if logits and len(logits) > 0 else -1
            if pred_index == label_remapped: correct_predictions_train += 1

            # --- Backward Pass ---
            grad = loss_fn.backward()
            grad = dense_output.backward(grad, lr)
            if dense1:
                grad_2d = [grad]
                grad = relu_dense1.backward(grad_2d)
                grad = grad[0]
                grad = dense1.backward(grad, lr)
            grad_3d = flatten.backward(grad)
            grad = grad_3d[0]
            grad = pool1.backward(grad)
            if conv2:
                grad = relu_conv2.backward(grad)
                grad_3d_from_c2 = conv2.backward(grad, lr)
                grad = grad_3d_from_c2[0]
            grad = relu_conv1.backward(grad)
            _ = conv1.backward(grad, lr)

        avg_train_loss = total_loss_train / len(current_epoch_shuffled_data)
        train_acc = correct_predictions_train / len(current_epoch_shuffled_data) * 100
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        print(f"Epoch {epoch + 1}/{epochs} DONE. Loss={avg_train_loss:.4f}, Acc={train_acc:.2f}%. Time: {time.time() - epoch_start_time:.2f}s")

    # === TESTING PHASE ===
    all_true_test, all_pred_test = [], []

    for img_vol_test, lbl_test in test_dataset:
        x_test = conv1.forward(img_vol_test)
        x_test = relu_conv1.forward(x_test)
        if conv2:
            x_c2_in_test = [x_test]
            x_test = conv2.forward(x_c2_in_test)
            x_test = relu_conv2.forward(x_test)
        x_test = pool1.forward(x_test)
        x_f_in_test = [x_test]
        x_flat_test = flatten.forward(x_f_in_test)
        if dense1:
            x_d1_test = dense1.forward(x_flat_test)
            x_d1_2d_test = [x_d1_test]
            x_r_d1_2d_test = relu_dense1.forward(x_d1_2d_test)
            x_test = x_r_d1_2d_test[0]
        else:
            x_test = x_flat_test
        logits_test = dense_output.forward(x_test)
        pred_idx_test = logits_test.index(max(logits_test)) if logits_test else -1
        all_true_test.append(lbl_test)
        all_pred_test.append(pred_idx_test)

    # === EVALUATION ===
    print("\n--- Final Evaluation ---")
    if all_true_test:
        acc_f = accuracy_score(all_true_test, all_pred_test)
        p, r, f1, _ = precision_recall_fscore_support(all_true_test, all_pred_test, average='weighted', zero_division=0)
        print(f"  Accuracy on test set: {acc_f * 100:.2f}% | P: {p:.2f} | R: {r:.2f} | F1: {f1:.2f}")

        actual_lbls = sorted(list(set(all_true_test)))
        if actual_lbls:
            pc, rc, f1c, sc = precision_recall_fscore_support(all_true_test, all_pred_test, labels=actual_lbls, average=None, zero_division=0)
            orig_map = {nl: ol for nl, ol in enumerate(target_mnist_classes)}
            for i, rl in enumerate(actual_lbls): print(
                f"  Cls {orig_map.get(rl, rl)} (map {rl}): P={pc[i]:.2f},R={rc[i]:.2f},F1={f1c[i]:.2f},S={sc[i]}")
    else:
        print("No test results.")

    if history['train_loss'] and history['train_acc']:
        epochs_range = range(1, epochs + 1)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, history['train_loss'], marker='o', label='Avg Train Loss')
        plt.title('Loss')
        plt.grid(True)
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, history['train_acc'], marker='o', label='Train Acc')
        plt.title('Accuracy')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("mnist_train_stats.png")
        if all_true_test:
            plt.figure(figsize=(6, 5))
            cm_lbls = list(range(num_classes))
            cm = confusion_matrix(all_true_test, all_pred_test, labels=cm_lbls)
            plt.imshow(cm, cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()
            tick_names = [str(target_mnist_classes[i]) if i < len(target_mnist_classes) else str(i) for i in cm_lbls]
            plt.xticks(cm_lbls, tick_names)
            plt.yticks(cm_lbls, tick_names)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            for r_cm in range(cm.shape[0]):
                for c_cm in range(cm.shape[1]):
                    plt.text(c_cm, r_cm, format(cm[r_cm, c_cm], 'd'), ha="center", va="center", color="white" if cm[r_cm, c_cm] > cm.max() / 2. else "black")
            plt.tight_layout()
            plt.savefig("mnist_confusion_matrix.png")
    else:
        print("\nNo history for plotting.")

if __name__ == '__main__':
    run_model()
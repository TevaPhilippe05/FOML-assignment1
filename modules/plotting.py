import matplotlib.pyplot as plt

def plot_metrics(metrics, cm_epochs=(), class_names=None):
    epochs = range(1, len(metrics["loss_train"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, metrics["accuracy_train"], label="Train")
    axes[0].plot(epochs, metrics["accuracy_val"], label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(epochs, metrics["loss_train"], label="Train")
    axes[1].plot(epochs, metrics["loss_val"], label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    for epoch in cm_epochs:
        index = epoch - 1
        if index < 0 or index >= len(metrics["confusion_matrices"]):
            print("No confusion matrix for epoch", epoch)
            continue

        cm = metrics["confusion_matrices"][index]
        plt.figure(figsize=(8, 8))
        plt.imshow(cm, cmap="Blues")
        if class_names is not None:
            plt.xticks(range(len(class_names)), class_names, rotation=90)
            plt.yticks(range(len(class_names)), class_names)
        plt.colorbar()
        plt.title("Confusion matrix - epoch " + str(epoch))
        plt.xlabel("Predicted class")
        plt.ylabel("True class")
        plt.show()
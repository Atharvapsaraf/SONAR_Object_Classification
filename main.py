from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def display_accuracy(target, predictions, labels, title):
    cm = confusion_matrix(target, predictions)
    cm_display = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots()
    cm_display.plot(ax=ax)
    ax.set_title(title)
    plt.show()

def evaluate(model, X_train, X_test, y_train, y_test, display_matrix=False):
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    if display_matrix:
        print(f'{model.__class__.__name__} Train Accuracy: {(y_pred_train == y_train).mean()}')
        display_accuracy(y_train, y_pred_train, ["Rock", "Mine"], "Confusion matrix")

    y_pred_test = model.predict(X_test)
    if display_matrix:
        print(f'{model.__class__.__name__} Test Accuracy: {(y_pred_test == y_test).mean()}')
        display_accuracy(y_test, y_pred_test, ["Rock", "Mine"], "Confusion matrix")

    return model.best_loss_

def main():
    # setting up train and test set
    inputs = np.genfromtxt('sonar.all-data.csv', usecols=range(60), delimiter=',')
    target = np.genfromtxt('sonar.all-data.csv', usecols=[60], dtype=np.dtype('U1'), delimiter=',') == 'M'
    X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2, random_state=42)

    num_layer_increase_losses = []
    num_layers = [(30,) * i for i in range(1, 10)]
    for hls in tqdm(num_layers, desc='Training Neural Networks with increasing number of layers'):
        num_layer_increase_losses.append(evaluate(MLPClassifier(
            random_state=0, hidden_layer_sizes=hls, batch_size=10, max_iter=50
        ), X_train, X_test, y_train, y_test))

    size_layer_increase_losses = []
    layer_sizes = [(10 * i, 10 * i) for i in range(1, 10)]
    for hls in tqdm(layer_sizes, 'Training Neural Networks with increasing size of layers'):
        size_layer_increase_losses.append(evaluate(MLPClassifier(
            random_state=0, hidden_layer_sizes=hls, batch_size=10, max_iter=50
        ), X_train, X_test, y_train, y_test))


    # fig, ax1 = plt.subplots()
    # ax1.plot(range(len(num_layer_increase_losses)), num_layer_increase_losses, color='blue', label='Increasing number of layers with 30 units')
    # ax1.xlabel('Number of Layers')
    # ax1.ylabel('Loss')
    # ax1.legend()
    # ax2 = ax1.twiny()
    # ax1.plot(range(len(size_layer_increase_losses)), size_layer_increase_losses, color='red', label='Increasing size of 2 layers by multiples of 10')
    # plt.show()


    # Your specific ticks for the x-axes
    num_layer_ticks = [len(t) for t in num_layers]  # Your x-ticks for ax1
    size_layer_ticks = layer_sizes  # Your x-ticks for ax2

    fig, ax1 = plt.subplots()

    # First plot and its x-axis
    print(num_layer_ticks)
    print(num_layer_increase_losses)
    ax1.plot(num_layer_ticks, num_layer_increase_losses, color='blue', label='Increasing number of layers with 30 units')
    ax1.set_xlabel('Number of Layers')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper left')

    # Create a second x-axis and plot the second line on the same y-axis
    ax2 = ax1.twiny()
    ax2.plot(size_layer_ticks, size_layer_increase_losses, color='red',
         label='Increasing size of 2 layers by multiples of 10')
    ax2.set_xlabel('Size of Layers')

    # Optionally, you may want to add this second legend. If you add this legend, consider adjusting the loc argument in both legends to avoid overlap.
    ax2.legend(loc='upper right')

    plt.show()

if __name__ == '__main__':
    main()

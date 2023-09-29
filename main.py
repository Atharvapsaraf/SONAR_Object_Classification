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
    for hls in tqdm([(30,) * i for i in range(1, 10)], desc='Training Neural Networks with increasing number of layers'):
        num_layer_increase_losses.append(evaluate(MLPClassifier(
            random_state=0, hidden_layer_sizes=hls, batch_size=10, max_iter=500
        ), X_train, X_test, y_train, y_test))

    size_layer_increase_losses = []
    for hls in tqdm([(10 * i, 10 * i) for i in range(1, 10)], 'Training Neural Networks with increasing size of layers'):
        size_layer_increase_losses.append(evaluate(MLPClassifier(
            random_state=0, hidden_layer_sizes=hls, batch_size=10, max_iter=500
        ), X_train, X_test, y_train, y_test))

    plt.plot(range(len(num_layer_increase_losses)), num_layer_increase_losses, color='blue', name='Increasing number of layers with 30 units')
    plt.plot(range(len(size_layer_increase_losses)), size_layer_increase_losses, color='blue', name='Increasing size of 2 layers by multiples of 10')
    plt.xlabel('Number of Layers')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()

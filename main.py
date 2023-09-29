from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def display_accuracy(target, predictions, labels, title):
    cm = confusion_matrix(target, predictions)
    cm_display = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots()
    cm_display.plot(ax=ax)
    ax.set_title(title)
    plt.show()

def evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    print(f'{model.__class__.__name__} Train Accuracy: {(y_pred_train == y_train).mean()}')
    display_accuracy(y_train, y_pred_train, ["Rock", "Mine"], "Confusion matrix")

    y_pred_test = model.predict(X_test)
    print(f'{model.__class__.__name__} Test Accuracy: {(y_pred_test == y_test).mean()}')
    display_accuracy(y_test, y_pred_test, ["Rock", "Mine"], "Confusion matrix")

def main():
    # setting up train and test set
    inputs = np.genfromtxt('sonar.all-data.csv', usecols=range(60), delimiter=',')
    target = np.genfromtxt('sonar.all-data.csv', usecols=[60], dtype=np.dtype('U1'), delimiter=',') == 'M'
    X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2, random_state=42)

    evaluate(MLPClassifier(
        random_state=0, hidden_layer_sizes=30, batch_size=10, max_iter=500, verbose=1
    ), X_train, X_test, y_train, y_test)

    evaluate(RandomForestClassifier(
        random_state=0, n_estimators=500,
    ), X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    main()

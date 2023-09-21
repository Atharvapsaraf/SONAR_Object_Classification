from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


def display_accuracy(target, predictions, labels, title):
    cm = confusion_matrix(target, predictions)
    cm_display = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots()
    cm_display.plot(ax=ax)
    ax.set_title(title)
    plt.show()


def main():
    inputs = np.genfromtxt('sonar.all-data.csv', usecols=range(60), delimiter=',')
    target = np.genfromtxt('sonar.all-data.csv', usecols=[60], dtype=np.dtype('U1'), delimiter=',') == 'M'
    X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2, random_state=42)

    classifier = MLPClassifier(random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(f'Accuracy: {(y_pred == y_test).mean()}')
    display_accuracy(y_test, y_pred, ["Rock", "Mine"], "Confusion matrix")

if __name__ == '__main__':
    main()



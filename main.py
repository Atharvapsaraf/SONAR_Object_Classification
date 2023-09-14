from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np

def main():
    inputs = np.genfromtxt('sonar.all-data.csv', usecols=range(60), delimiter=',')
    target = np.genfromtxt('sonar.all-data.csv', usecols=[60], dtype=np.dtype('U1'), delimiter=',') == 'R'
    X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2, random_state=42)

    classifier = MLPClassifier(random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(f'Accuracy: {(y_pred == y_test).mean()}')

if __name__ == '__main__':
    main()

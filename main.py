from CNN import CNN
from utils import graph_history


def main():
    cnn = CNN()
    cnn.visualize_model()
    history = cnn.train(epochs=30)
    #cnn.confusion_matrix(normalize=True)
    #graph_history(history)


if __name__ == "__main__":
    main()

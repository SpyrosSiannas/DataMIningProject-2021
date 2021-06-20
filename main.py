from src.nn import NeuralNetwork

if __name__ == "__main__":
    nn = NeuralNetwork()
    choice = int(input("Do you want to:\n1. Train the Neural Network\n2. Load the Neural Network\n"))
    if choice == 1:
        nn.train_nn_model()
    elif choice == 2:
        nn.loadmodel()
    nn.test_nn()
    print("Model loaded, proceeding to test")
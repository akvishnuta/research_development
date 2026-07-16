from deeplearning.baseline.baseline import DataExtractor 
import numpy as np
from sklearn.model_selection import train_test_split
import time

class MLP:
    def __init__(self, architecture, learning_rate=0.01, n_iterations=1000):
        self.architecture = architecture
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.loss_history = []

    
    def initialize_parameters(self):
        self.parameters = {}

        #Initialize parameter with Xavier's initialization
        for i in range(1, len(self.architecture)):
            limit = (6/(self.architecture[i-1] + self.architecture[i]))
            self.parameters[f"W{i}"] = np.random.uniform(
                -limit, limit, (self.architecture[i-1], self.architecture[i])
            )
            #initilize 1 bias per neuron in the given layer
            self.parameters[f"b{i}"] = np.zeros((1, self.architecture[i]))

        
        print("Initial paramters: ")
        print(self.parameters)


    #Using Leaky relu to avoid gradient vanishing problem
    def relu(self, Z):
      return np.maximum(0.01 * Z, Z)

    def relu_derivative(self, Z):
      return np.where(Z > 0, 1, 0.01)


    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    def forward_propagation(self, X):
        self.cache = {"A0": X}
        L = len(self.architecture) - 1

        #Hidden Layers calculation
        for i in range(1, L):
            #z=x*w+b
            Z = self.cache[f"A{i-1}"] @ self.parameters[f"W{i}"] + self.parameters[f"b{i}"]
            self.cache[f"Z{i}"] = Z
            #Apply relu activation function for all hidden layers
            self.cache[f"A{i}"] = self.relu(Z)


        #Output Layer calculation
        ZL = self.cache[f"A{L-1}"] @ self.parameters[f"W{L}"] + self.parameters[f"b{L}"]
        #Apply sigmoid activation function for output layer
        self.cache[f"A{L}"] = self.sigmoid(ZL)

        #return the output
        return self.cache[f"A{L}"]

    def backward_propagation(self, X, y, y_hat):
        m = X.shape[0]
        grads = {}
        L = len(self.architecture) - 1

        #Loss=−[ylog(y_hat​)+(1−y)log(1−y_hat​)]
        #derivative of the loss with respect to the pre-activation ZL => d(AL)/d(ZL)
        #simplifies to AL-y

        dA = y_hat - y.reshape(-1,1) #y is reshaped to a column vector

        for i in reversed(range(1, L+1)):
            grads[f"dW{i}"] = (1/m) * self.cache[f"A{i-1}"].T @ dA
            grads[f"db{i}"] = (1/m) * np.sum(dA, axis=0, keepdims=True)

            if i > 1:
                dA = (dA @ self.parameters[f"W{i}"].T) * self.relu_derivative(self.cache[f"Z{i-1}"])

        return grads
    
    def fit(self, X, y, enable_early_stopping=True):
        self.initialize_parameters()

        patience = 10
        min_delta = 1e-5
        best_loss = float("inf")
        patience_counter = 0

        n_samples = X.shape[0]

        for epoch in range(self.n_iterations):
            epoch_loss = 0

            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Using Stochastic Gradient Descent to Train the model
            for i in range(n_samples):
                x_i = X_shuffled[i].reshape(1, -1)
                y_i = np.array([[y_shuffled[i]]])

                y_hat = self.forward_propagation(x_i)

                #so as to avoid log(0) = infinity in the later steps
                y_hat = np.clip(y_hat, 1e-8, 1 - 1e-8)

                loss = - (y_i * np.log(y_hat) + (1 - y_i) * np.log(1 - y_hat))
                epoch_loss += loss.item()

                grads = self.backward_propagation(x_i, y_i, y_hat)

                for l in range(1, len(self.architecture)):
                    self.parameters[f"W{l}"] -= self.lr * grads[f"dW{l}"]
                    self.parameters[f"b{l}"] -= self.lr * grads[f"db{l}"]

            epoch_loss /= n_samples
            self.loss_history.append(epoch_loss)

            # Implemented Early stopping
            if enable_early_stopping:
                if epoch_loss < best_loss - min_delta:
                    best_loss = epoch_loss
                    patience_counter = 0
                    best_weights = {k: v.copy() for k, v in self.parameters.items()}
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    self.parameters = best_weights
                    break

        return self

    def predict(self, X):
        y_hat = self.forward_propagation(X)
        return (y_hat >= 0.5).astype(int)



def main():
    FILE_PATH = "./deeplearning/data/wisconsin_original.csv"
    COLUMN_NAMES = [
        "id",
        "clump_thickness",
        "cell_size_uniformity",
        "cell_shape_uniformity",
        "marginal_adhesion",
        "epithelial_cell_size",
        "bare_nuclei",
        "bland_chromatin",
        "normal_nucleoli",
        "mitoses",
        "class"
    ]
    dataExtractor = DataExtractor(FILE_PATH, COLUMN_NAMES)
    df = dataExtractor.read_csv()
    X,y = dataExtractor.preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    architecture = [X_train.shape[1], 16,8,1]

    print("Training MLP...")
    mlp_start_time = time.time()

    mlp = MLP(architecture=architecture)



    architecture=[X_train.shape[1], 16, 8, 1]

    enable_early_stopping = True
    mlp.fit(X_train, y_train, enable_early_stopping)

    mlp_training_time = time.time() - mlp_start_time

    print(f"✓ MLP training completed in {mlp_training_time:.2f}s")
    print(f"✓ Loss decreased from {mlp.loss_history[0]:.4f} to {mlp.loss_history[-1]:.4f}")



    y_pred_mlp = mlp.predict(X_test)


    mlp_initial_loss = mlp.loss_history[0]
    mlp_final_loss = mlp.loss_history[-1]





if __name__ == "__main__":
    main()

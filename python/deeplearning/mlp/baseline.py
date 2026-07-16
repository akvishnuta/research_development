import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
class BaseLineModel:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.loss_history = []

    
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    

    def relu(self, z):
        return z if z>0 else 0
    
    def fit(self, X, y):
        m,n = X.shape
        self.w = np.zeros(n)
        self.b = 0

        for epoch in range(self.n_iterations):
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0

            for i in range(m):
                x_i = X_shuffled[i]
                y_i = y_shuffled[i]

                z = np.dot(x_i, self.w) + self.b

                y_hat = self.sigmoid(z)

                #clipping y_hat so as to avoid log(1) = 0 in later steps
                y_hat = np.clip(y_hat, 1e-8, 1-1e-8)

                #calculate loss
                loss = -(y_i * np.log(y_hat) + (1-y_i)*np.log(1-y_hat))

                epoch_loss +=loss

                dw = (y_hat - y_i) * x_i
                db = (y_hat - y_i)


                #Parameter update
                self.w -= self.lr * dw
                self.b -= self.lr * db

            self.loss_history.append(epoch_loss/m)    
        return self
    
    def predict(self, X):
        z = np.dot(X, self.w) + self.b

        y_hat = (self.sigmoid(z)>0.5).astype(int)
        return y_hat



class DataExtractor:
    def __init__(self, filepath, column_names):
        self.filepath = filepath
        self.column_names = column_names
        

    def read_csv(self):
        df = pd.read_csv(self.filepath,
                    header=None,
                    names=self.column_names,
                    na_values="?",
                    usecols=range(len(self.column_names)))
        print(f"Initial shape : {df.shape}")
        print(df.head())
        return df
    
    def preprocess(self, df):
        df.dropna(inplace=True)
        print(f"Shape after dropping missing values : {df.shape}")
        df.drop(columns=["id"], inplace=True)
        print(f"Shape after dropping id : {df.shape}")

        y= df["class"].values
        X= df.drop(columns=["class"]).values

        # Convert labels: 2 -> 0 (Benign), 4 -> 1 (Malignant)
        y=np.where(y==4, 1, 0)

        #Normalize
        X=(X-X.mean())/(X.std() + 1e-8)

        print("X shape : ", X.shape)
        print("y distribution", np.bincount(y))

        return X,y
    

    

    



def main():
    FILE_PATH = "../data/wisconsin_original.csv"
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

    base_line = BaseLineModel()
    
    # Train baseline model
    print("Training baseline model...")
    baseline_start = time.time()
    
    base_line.fit(X_train, y_train)

    
    baseline_predictions = base_line.predict(X_test)



    baseline_training_time = time.time() - baseline_start
    print(f"✓ Baseline training completed in {baseline_training_time:.2f}s")
    print(f"✓ Loss decreased from {base_line.loss_history[0]:.4f} to {base_line.loss_history[-1]:.4f}")

    # Store loss explicitly
    baseline_initial_loss = base_line.loss_history[0]
    baseline_final_loss = base_line.loss_history[-1]


if __name__ == "__main__":
    main()

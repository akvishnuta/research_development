import numpy as np
import pandas as pd
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
        
        return
    



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


    base_line = BaseLineModel(FILE_PATH, COLUMN_NAMES)
    


if __name__ == "__main__":
    main()

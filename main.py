import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.impute import LinearRegression


if __name__ == "__main__":
    df = pd.read_csv("dataset/healthcare-dataset-stroke-data/healthcare-dataset-stroke-data.csv")
    dataset3 = df
    x = dataset3.age.values
    x = x.reshape(x.shape[0],1)
    y = dataset3.bmi

    y = y.values.reshape(y.shape[0], 1)

    lreg = LinearRegression()
    lreg.fit(x , y)

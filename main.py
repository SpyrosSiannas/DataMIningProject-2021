import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    df = pd.read_csv("dataset/healthcare-dataset-stroke-data/healthcare-dataset-stroke-data.csv")
    df.plot(x="id", y=["bmi"])
    plt.show()

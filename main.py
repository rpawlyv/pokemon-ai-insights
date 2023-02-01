import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


df = pd.read_csv("Pokemon_Data.csv")
stats = df.columns.to_list()[3: 9]

for feature in stats:
    x = pd.DataFrame(df[feature])
    y = pd.DataFrame(df["Stat Total"])
    lin = LinearRegression()
    lin.fit(x, y)
    ax = df.plot.scatter(x=feature, y="Stat Total", alpha=0.5)
    ax.plot(x, lin.predict(x), c='r')
    print(f"The correlation between {feature} and BST is {lin.score(x, y)}")

plt.show()
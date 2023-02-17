import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler


def baseStatRegression(df):
    stats = df.columns.to_list()[3: 9]
    y = pd.DataFrame(df["Stat Total"])

    for feature in stats:
        x = pd.DataFrame(df[feature])
        lin = LinearRegression()
        lin.fit(x, y)
        ax = df.plot.scatter(x=feature, y="Stat Total", alpha=0.5)
        ax.plot(x, lin.predict(x), c='r')
        print(f"The correlation between {feature} and BST is {lin.score(x, y)}")

    plt.show()

df = pd.read_csv("Pokemon_Data.csv")
evo_df = df[df['Is Fully Evolved']]
evo_stats_df = evo_df.loc[evo_df["Is Fully Evolved"] & ~evo_df["Is Legendary"], df.columns.to_list()[3: 9]]

scaler = MinMaxScaler()
scaler.fit(evo_stats_df)
X = scaler.transform(evo_stats_df)
kmeans = KMeans(n_clusters=5, init="k-means++",)
kmeans.fit(X)
clusters=pd.DataFrame(X,columns=evo_stats_df.columns)
clusters['label']=kmeans.labels_
polar=clusters.groupby("label").mean().reset_index()
polar=pd.melt(polar,id_vars=["label"])
fig4 = px.line_polar(polar, r="value", theta="variable", color="label", line_close=True,height=800,width=1400)
fig4.show()


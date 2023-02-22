from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import numpy as np
import pandas as pd

df = pd.read_csv("Pokemon_Data.csv")
evo_df = df[df['Is Fully Evolved']]
evo_stats_df = evo_df.loc[evo_df["Is Fully Evolved"] & ~evo_df["Is Legendary"], df.columns.to_list()[3: 9]]

X = evo_stats_df
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
inertia = []
for i in range(1,11):
    kmeans = KMeans(
        n_clusters=i, init="k-means++",
        n_init=10,
        tol=1e-04, random_state=42
    )
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)
fig = go.Figure(data=go.Scatter(x=np.arange(1,11),y=inertia))
fig.update_layout(title="Inertia vs Cluster Number",xaxis=dict(range=[0,11],title="Cluster Number"),
                  yaxis={'title':'Inertia'},
                 annotations=[
        dict(
            x=3,
            y=inertia[2],
            xref="x",
            yref="y",
            text="Elbow!",
            showarrow=True,
            arrowhead=7,
            ax=20,
            ay=-40
        )
    ])
fig.show()
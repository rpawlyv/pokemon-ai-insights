from sklearn.cluster import KMeans
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def poke_stat_clustering(df):
    scaler = MinMaxScaler()
    scaler.fit(df)
    X = scaler.transform(df)
    kmeans = KMeans(n_clusters=3, init="k-means++", )
    kmeans.fit(X)
    clusters = pd.DataFrame(X, columns=df.columns)
    clusters['label'] = kmeans.labels_
    polar = clusters.groupby("label").mean().reset_index()
    polar = pd.melt(polar, id_vars=["label"])
    fig4 = px.line_polar(polar, r="value", theta="variable", color="label", line_close=True, height=800, width=1400)
    fig4.show()
    pie=clusters.groupby('label').size().reset_index()
    pie.columns=['label','value']
    fig5 = px.pie(pie,values='value',names='label',color=['blue','red','green'])
    fig5.show()

df = pd.read_csv("Pokemon_Data.csv")
evo_df = df[df['Is Fully Evolved']]
evo_stats_df = evo_df.loc[evo_df["isFullyEvolved"] & ~evo_df["isLegendary"], df.columns.to_list()[3: 9]]

poke_stat_clustering(evo_stats_df)
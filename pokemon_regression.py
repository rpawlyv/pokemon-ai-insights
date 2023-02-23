import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression



def poke_stat_regression(df, yStat):
    y = pd.DataFrame(df[yStat])

    for feature in df:
        x = pd.DataFrame(df[feature])
        lin = LinearRegression()
        lin.fit(x, y)
        ax = df.plot.scatter(x=feature, y=yStat, alpha=0.5)
        ax.plot(x, lin.predict(x), c='r')
        print(f"The correlation between {feature} and {yStat} is {lin.score(x, y)}")

    plt.show()


df = pd.read_csv("Pokemon_Data.csv")
evo_df = df[df['Is Fully Evolved']]
evo_stats_df = evo_df.loc[evo_df["Is Fully Evolved"] & ~evo_df["Is Legendary"], df.columns.to_list()[2: 9]]

for stat in evo_stats_df:
    poke_stat_regression(evo_stats_df, stat)
    print("")

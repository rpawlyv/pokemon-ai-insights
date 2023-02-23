from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pandas as pd




df = pd.read_csv("Pokemon_Data.csv")
df["Is Fully Evolved"] = df["Is Fully Evolved"].astype(int)


X = df[["HP", "Attack", "Defense", "Sp.Atk", "Sp.Def", "Speed", "Stat Total"]]
Y = df["Is Fully Evolved"]

clf = RandomForestClassifier(random_state=42)

clf.fit(X, Y)

for index, value in enumerate(clf.predict(X), 0):
    if list(Y)[index] != value:
        print(f"{list(df[index:index+1].Name)} has an error. Its Fully Evolved Data is {Y[index:index+1].values}")
        print(f"{list(Y)[index]} should be the right answer but the model got {value} instead")
        print(f"Probability was the following: {clf.predict_proba(X[index:index+1])}")
        print("")
print(f"The score without splitting data is {clf.score(X, Y)}")
print("")

#split test

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

clf.fit(X_train, Y_train)



for index, value in enumerate(clf.predict(X_test), 0):
    if list(Y_test)[index] != value:
        print(f"{list(df[index:index+1].Name)} has an error. Its Fully Evolved Data is {Y_test[index:index+1].values}")
        print(f"{list(Y_test)[index]} should be the right answer but the model got {value} instead")
        print(f"Probability was the following: {clf.predict_proba(X_test[index:index+1])}")
        print("")
print("")
print(f"The score without splitting data is {clf.score(X_test, Y_test)}")

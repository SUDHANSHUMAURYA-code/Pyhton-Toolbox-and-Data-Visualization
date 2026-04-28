import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, ttest_ind

df = pd.read_csv(r"C:\Users\Sudhanshu\Downloads\python datatset.csv")

df.columns = df.columns.str.strip().str.replace(" ", "_")

df = df.drop_duplicates()
df.drop(columns=[c for c in ["Country","Year"] if c in df.columns], inplace=True, errors="ignore")

num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

cat_cols = df.select_dtypes(include="object").columns
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

def pick(col_keywords):
    for c in df.columns:
        if all(k.lower() in c.lower() for k in col_keywords):
            return c
    return None

col_lit = pick(["women","literat"])
col_birth = pick(["institutional","birth"])
col_stunt = pick(["stunt"])
col_res = pick(["residence"])

plt.figure()
sns.histplot(df[col_lit].dropna(), kde=True)
plt.show()

plt.figure()
sns.scatterplot(x=col_lit, y=col_birth, data=df)
plt.show()

if col_res and col_stunt:
    plt.figure()
    sns.boxplot(x=col_res, y=col_stunt, data=df)
    plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
plt.show()

stat, p = shapiro(df[col_lit].dropna())

if p > 0.05:
    print("Normal Data")
else:
    print("Not Normal")

if col_res:
    rural = df[df[col_res].str.contains("Rural", case=False, na=False)][col_lit]
    urban = df[df[col_res].str.contains("Urban", case=False, na=False)][col_lit]

    t, p = ttest_ind(rural.dropna(), urban.dropna())

    print("p-value:", p)

    if p < 0.05:
        print("Significant Difference")
    else:
        print("No Significant Difference")

print("Urban > Rural literacy & healthcare")
print("Rural > malnutrition")
print("Literacy ↑ → Births ↑")
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = pd.read_csv("student-columns.csv")
df.columns = df.columns.str.strip()

print("Original Data:")
print(df.head())
print("\nMissing Values:")
print(df.isnull().sum())

df["school"] = df["school"].fillna(df["school"].mode()[0])
df["sex"] = df["sex"].fillna(df["sex"].mode()[0])
for col in ["age", "Medu", "Fedu"]:
    df[col] = df[col].fillna(df[col].median())

print("\nCleaned Data:")
print(df.head())
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

minmax = MinMaxScaler()
df["AGE_MINMAX"] = minmax.fit_transform(df[["age"]])

standard = StandardScaler()
df["AGE_ZSCORE"] = standard.fit_transform(df[["age"]])

print("\nScaled Data:")
print(df[["age", "AGE_MINMAX", "AGE_ZSCORE"]].head())
print("\nStatistics:")
print(df[["age", "Medu", "Fedu"]].describe())

df_clean = df.dropna()
df_clean.to_csv("student_columns_cleaned.csv", index=False)
print("\nCleaned data saved successfully.")


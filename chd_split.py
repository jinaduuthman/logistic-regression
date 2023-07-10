import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("framingham.csv", index_col=False)
print(f"Read {df.shape[0]} rows")

# Drop rows with missing values
## Your code here
df_Cleaned = df.dropna(axis=0, inplace=False)

print(f"Using {df.shape[0]} rows")

# Split into training and testing dataframes
## Your code here
split_list = train_test_split(df_Cleaned, test_size=0.2)
train_df = split_list[0]
test_df = split_list[1]


# Write out each as a CSV
## Your code here
# Export train_df to CSV
train_df.to_csv(index=False)
train_df.to_csv("train.csv")
# df.to_csv('df.csv')

# Export test_df to CSV
test_df.to_csv(index=False)
test_df.to_csv("test.csv")

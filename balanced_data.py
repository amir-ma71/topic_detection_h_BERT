from sklearn.utils import resample
import pandas as pd
import numpy as np

label_list = ["entertainment", "sport", "news", "technology", "health", "economy"]

def add_id(label):
    if label == "entertainment":
        return 0
    elif label == "sport":
        return 1
    elif label == "news":
        return 2
    elif label == "tech":
        return 3
    elif label == "health":
        return 4
    elif label == "economy":
        return 5
    else:
        return -1

# Read dataset
df = pd.read_csv('./src/input_data/final.csv', quoting=1, encoding="utf-8", sep="~")

# print(df['label'].value_counts())

# outputs:
# news             13319
# sport             7455
# health            3496
# entertainment     3300
# economy           2389
# tech              1863

df['unique_id'] = df["label"].apply(lambda x: add_id(x))
df = df[df.unique_id != -1]
labels =df['unique_id'].value_counts()
print(labels)

count_label = {}
for idx, name in enumerate(df['label'].value_counts().index.tolist()):
    count_label[name] = df['label'].value_counts()[idx]


# Separate majority and minority classes
df_0 = df[df.unique_id == 0]
df_1 = df[df.unique_id == 1]
df_2 = df[df.unique_id == 2]
df_3 = df[df.unique_id == 3]
df_4 = df[df.unique_id == 4]
df_5 = df[df.unique_id == 5]


# Upsample minority class
df_sampled_4 = resample(df_4,
                                 replace=True,  # sample with replacement
                                 n_samples=7455,  # to match majority class
                                 random_state=123)  # reproducible results
df_sampled_0 = resample(df_0,
                                 replace=True,  # sample with replacement
                                 n_samples=7455,  # to match majority class
                                 random_state=123)  # reproducible results
df_sampled_5 = resample(df_5,
                                 replace=True,  # sample with replacement
                                 n_samples=7455,  # to match majority class
                                 random_state=123)  # reproducible results
df_sampled_3 = resample(df_3,
                                 replace=True,  # sample with replacement
                                 n_samples=7455,  # to match majority class
                                 random_state=123)  # reproducible results
df_sampled_2 = resample(df_2,
                                 replace=False,  # sample with replacement
                                 n_samples=7455,  # to match majority class
                                 random_state=123)  # reproducible results

# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_1, df_sampled_4, df_sampled_0, df_sampled_5, df_sampled_3, df_sampled_2])

df_upsampled.to_csv('./src/input_data/final_balance.csv', encoding="utf-8", sep="~", quoting=1, index=False)




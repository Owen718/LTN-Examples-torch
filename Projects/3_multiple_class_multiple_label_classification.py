import torch
import pandas as pd

df = pd.read_csv("Projects/datasets/crabs.dat", sep=" ", skipinitialspace=True)
df = df.sample(frac=1)  # shuffle dataset
df = df.replace({'B': 0, 'O': 1, 'M': 2, 'F': 3})

features = torch.tensor(df[['FL', 'RW', 'CL', 'CW', 'BD']].to_numpy())
labels_sex = torch.tensor(df['sex'].to_numpy())
labels_color = torch.tensor(df['sp'].to_numpy())

train_data = features[:160].float()
test_data = features[160:].float()
train_sex_labels = labels_sex[:160].long()
test_sex_labels = labels_sex[160:].long()
train_color_labels = labels_color[:160].long()
test_color_labels = labels_color[160:].long()


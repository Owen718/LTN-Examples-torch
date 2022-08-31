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

import ltn

# we define the constants
l_blue = ltn.Constant(torch.tensor([1, 0, 0, 0]))
l_orange = ltn.Constant(torch.tensor([0, 1, 0, 0]))
l_male = ltn.Constant(torch.tensor([0, 0, 1, 0]))
l_female = ltn.Constant(torch.tensor([0, 0, 0, 1]))

# we define predicate P
class MLP(torch.nn.Module):
    """
    This model returns the logits for the classes given an input example. It does not compute the softmax, so the output
    are not normalized.
    This is done to separate the accuracy computation from the satisfaction level computation. Go through the example
    to understand it.
    """
    def __init__(self, layer_sizes=(5, 16, 16, 8, 4)):
        super(MLP, self).__init__()
        self.elu = torch.nn.ELU()
        self.dropout = torch.nn.Dropout(0.2)
        self.linear_layers = torch.nn.ModuleList([torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i])
                                                  for i in range(1, len(layer_sizes))])

    def forward(self, x, training=False):
        """
        Method which defines the forward phase of the neural network for our multi class classification task.
        In particular, it returns the logits for the classes given an input example.

        :param x: the features of the example
        :param training: whether the network is in training mode (dropout applied) or validation mode (dropout not applied)
        :return: logits for example x
        """
        for layer in self.linear_layers[:-1]:
            x = self.elu(layer(x))
            if training:
                x = self.dropout(x)
        logits = self.linear_layers[-1](x)
        return logits


class LogitsToPredicate(torch.nn.Module):
    """
    This model has inside a logits model, that is a model which compute logits for the classes given an input example x.
    The idea of this model is to keep logits and probabilities separated. The logits model returns the logits for an example,
    while this model returns the probabilities given the logits model.

    In particular, it takes as input an example x and a class label l. It applies the logits model to x to get the logits.
    Then, it applies a softmax function to get the probabilities per classes. Finally, it returns only the probability related
    to the given class l.
    """
    def __init__(self, logits_model):
        super(LogitsToPredicate, self).__init__()
        self.logits_model = logits_model
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, l, training=False):
        logits = self.logits_model(x, training=training)
        probs = self.sigmoid(logits)
        out = torch.sum(probs * l, dim=1)
        return out

mlp = MLP()
P = ltn.Predicate(LogitsToPredicate(mlp))

# we define the connectives, quantifiers, and the SatAgg
Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
And = ltn.Connective(ltn.fuzzy_ops.AndProd())
Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
SatAgg = ltn.fuzzy_ops.SatAgg()


from sklearn.metrics import accuracy_score
import numpy as np

class DataLoader(object):
    def __init__(self,
                 data,
                 labels,
                 batch_size=1,
                 shuffle=True):
        self.data = data
        self.labels_sex = labels[0]
        self.labels_color = labels[1]
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(self.data.shape[0] / self.batch_size))

    def __iter__(self):
        n = self.data.shape[0]
        idxlist = list(range(n))
        if self.shuffle:
            np.random.shuffle(idxlist)

        for _, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            data = self.data[idxlist[start_idx:end_idx]]
            labels_sex = self.labels_sex[idxlist[start_idx:end_idx]]
            labels_color = self.labels_color[idxlist[start_idx:end_idx]]

            yield data, labels_sex, labels_color


# define metrics for evaluation of the model

# it computes the overall satisfaction level on the knowledge base using the given data loader (train or test)
def compute_sat_level(loader):
    mean_sat = 0
    for data, labels_sex, labels_color in loader:
        x = ltn.Variable("x", data)
        x_blue = ltn.Variable("x_blue", data[labels_color == 0])
        x_orange = ltn.Variable("x_orange", data[labels_color == 1])
        x_male = ltn.Variable("x_male", data[labels_sex == 2])
        x_female = ltn.Variable("x_female", data[labels_sex == 3])
        mean_sat += SatAgg(
            Forall(x_blue, P(x_blue, l_blue)),
            Forall(x_orange, P(x_orange, l_orange)),
            Forall(x_male, P(x_male, l_male)),
            Forall(x_female, P(x_female, l_female)),
            Forall(x, Not(And(P(x, l_blue), P(x, l_orange)))),
            Forall(x, Not(And(P(x, l_male), P(x, l_female))))
        )
    mean_sat /= len(loader)
    return mean_sat

# it computes the overall accuracy of the predictions of the trained model using the given data loader
# (train or test)
def compute_accuracy(loader, threshold=0.5):
    mean_accuracy = 0.0
    for data, labels_sex, labels_color in loader:
        predictions = mlp(data).detach().numpy()
        labels_male = (labels_sex == 2)
        labels_female = (labels_sex == 3)
        labels_blue = (labels_color == 0)
        labels_orange = (labels_color == 1)
        onehot = np.stack([labels_blue, labels_orange, labels_male, labels_female], axis=-1).astype(np.int32)
        predictions = predictions > threshold
        predictions = predictions.astype(np.int32)
        nonzero = np.count_nonzero(onehot - predictions, axis=-1).astype(np.float32)
        multilabel_hamming_loss = nonzero / predictions.shape[-1]
        mean_accuracy += np.mean(1 - multilabel_hamming_loss)

    return mean_accuracy / len(loader)

# create train and test loader
train_loader = DataLoader(train_data, (train_sex_labels, train_color_labels), 64, shuffle=True)
test_loader = DataLoader(test_data, (test_sex_labels, test_color_labels), 64, shuffle=False)



Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())

def phi1(features):
    x = ltn.Variable("x", features)
    return Forall(x, Implies(P(x, l_blue), Not(P(x, l_orange))), p=5)

def phi2(features):
    x = ltn.Variable("x", features)
    return Forall(x, Implies(P(x, l_blue), P(x, l_orange)), p=5)

def phi3(features):
    x = ltn.Variable("x", features)
    return Forall(x, Implies(P(x, l_blue), P(x, l_male)), p=5)

# it computes the satisfaction level of a formula phi using the given data loader (train or test)
def compute_sat_level_phi(loader, phi):
    mean_sat = 0
    for features, _, _ in loader:
        mean_sat += phi(features).value
    mean_sat /= len(loader)
    return mean_sat

optimizer = torch.optim.Adam(P.parameters(), lr=0.001)

for epoch in range(500):
    train_loss = 0.0
    for batch_idx, (data, labels_sex, labels_color) in enumerate(train_loader):
        optimizer.zero_grad()
        # we ground the variables with current batch data
        x = ltn.Variable("x", data)
        x_blue = ltn.Variable("x_blue", data[labels_color == 0])
        x_orange = ltn.Variable("x_orange", data[labels_color == 1])
        x_male = ltn.Variable("x_male", data[labels_sex == 2])
        x_female = ltn.Variable("x_female", data[labels_sex == 3])
        sat_agg = SatAgg(
            Forall(x_blue, P(x_blue, l_blue)),
            Forall(x_orange, P(x_orange, l_orange)),
            Forall(x_male, P(x_male, l_male)),
            Forall(x_female, P(x_female, l_female)),
            Forall(x, Not(And(P(x, l_blue), P(x, l_orange)))),
            Forall(x, Not(And(P(x, l_male), P(x, l_female))))
        )
        loss = 1. - sat_agg
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss = train_loss / len(train_loader)

    # we print metrics every 20 epochs of training
    if epoch % 20 == 0:
        print(" epoch %d | loss %.4f | Train Sat %.3f | Test Sat %.3f | Train Acc %.3f | Test Acc %.3f | "
                        "Test Sat Phi 1 %.3f | Test Sat Phi 2 %.3f | Test Sat Phi 3 %.3f " %
              (epoch, train_loss, compute_sat_level(train_loader),
                        compute_sat_level(test_loader),
                        compute_accuracy(train_loader), compute_accuracy(test_loader),
                        compute_sat_level_phi(test_loader, phi1), compute_sat_level_phi(test_loader, phi2),
                        compute_sat_level_phi(test_loader, phi3)))
import torch
import pandas as pd
from ltn_utils.utils import ExclusionPairedClause,IsA_PairedClause
train_data = pd.read_csv("Projects/datasets/iris_training.csv")
test_data = pd.read_csv("Projects/datasets/iris_test.csv")

train_labels = train_data.pop("species")
test_labels = test_data.pop("species")

train_data = torch.tensor(train_data.to_numpy()).float()
test_data = torch.tensor(test_data.to_numpy()).float()
train_labels = torch.tensor(train_labels.to_numpy()).long()
test_labels = torch.tensor(test_labels.to_numpy()).long()

import ltn

# we define the constants
l_A = ltn.Constant(torch.tensor([1, 0, 0]),ConstantName='l_A')
l_B = ltn.Constant(torch.tensor([0, 1, 0]),ConstantName='l_B')
l_C = ltn.Constant(torch.tensor([0, 0, 1]),ConstantName='l_C')

# we define predicate P
class MLP(torch.nn.Module):
    """
    This model returns the logits for the classes given an input example. It does not compute the softmax, so the output
    are not normalized.
    This is done to separate the accuracy computation from the satisfaction level computation. Go through the example
    to understand it.
    """
    def __init__(self, layer_sizes=(4, 16, 16, 8, 3)):
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
    def __init__(self):
        super(LogitsToPredicate, self).__init__()
      #  self.logits_model = logits_model
        self.softmax = torch.nn.Softmax(dim=1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, logits, l, training=False):
        '''
        l: [B,C]
        probs:[B,C]
        out:[B] ->  out[x] in [0,1] -> discrete bool result

        '''

        
        #logits = self.logits_model(x, training=training)
        #probs = self.softmax(logits) #[B,C]
        probs = self.sigmoid(logits)
        
        out = torch.sum(probs * l, dim=1)
        return out



net = MLP()
P = ltn.Predicate(LogitsToPredicate())
# we define the connectives, quantifiers, and the SatAgg
Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
And = ltn.Connective(ltn.fuzzy_ops.AndProd())
# we define the connectives, quantifiers, and the SatAgg
Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
SatAgg = ltn.fuzzy_ops.SatAgg()


from sklearn.metrics import accuracy_score
import numpy as np

# this is a standard PyTorch DataLoader to load the dataset for the training and testing of the model
class DataLoader(object):
    def __init__(self,
                 data,
                 labels,
                 batch_size=1,
                 shuffle=True):
        self.data = data
        self.labels = labels
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
            labels = self.labels[idxlist[start_idx:end_idx]]

            yield data, labels


# define metrics for evaluation of the model

# it computes the overall satisfaction level on the knowledge base using the given data loader (train or test)
def compute_sat_level(loader,Clause_Total):
    mean_sat = 0
    for data, labels in loader:
        x_A = ltn.Variable("x_A", data[labels == 0])
        x_B = ltn.Variable("x_B", data[labels == 1])
        x_C = ltn.Variable("x_C", data[labels == 2])
        mean_sat += SatAgg(
        Clause_Total
        )
    mean_sat /= len(loader)
    return mean_sat

# it computes the overall accuracy of the predictions of the trained model using the given data loader
# (train or test)
def compute_accuracy(loader):
    mean_accuracy = 0.0
    for data, labels in loader:
        predictions = net(data).detach().numpy()
        predictions = np.argmax(predictions, axis=1)
        mean_accuracy += accuracy_score(labels, predictions)

    return mean_accuracy / len(loader)

# create train and test loader
train_loader = DataLoader(train_data, train_labels, 64, shuffle=True)
test_loader = DataLoader(test_data, test_labels, 64, shuffle=False)


optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

for epoch in range(2000):
    train_loss = 0.0
    if epoch == 1: ## for debug
        print('') 
    if epoch == 400: ## for debug
        print('') 
    
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        # we ground the variables with current batch data

        preds = net(data)

        x = ltn.Variable("x",preds)
        x_A = ltn.Variable("x_A", preds[labels == 0]) # class A examples
        x_B = ltn.Variable("x_B", preds[labels == 1]) # class B examples
        x_C = ltn.Variable("x_C", preds[labels == 2]) # class C examples

        
        Clause_isa,isa_str = IsA_PairedClause(P,[x_A,x_B,x_C],[l_A,l_B,l_C])
        Clause_Exclusion,exc_str = ExclusionPairedClause(P,x,[l_A,l_B,l_C])

        
        Clause_Total = Clause_isa+Clause_Exclusion
        sat_agg = SatAgg(
            Clause_Total
         )
        loss = 1. - sat_agg
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss = train_loss / len(train_loader)

    # we print metrics every 20 epochs of training
    if epoch % 20 == 0:
        print(" epoch %d | loss %.4f | Train Sat %.3f | Test Sat %.3f | Train Acc %.3f | Test Acc %.3f"
              %(epoch, train_loss, compute_sat_level(train_loader,Clause_Total), compute_sat_level(test_loader,Clause_Total),
                    compute_accuracy(train_loader), compute_accuracy(test_loader)))
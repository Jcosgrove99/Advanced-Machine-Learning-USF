import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def proc_col(col, train_col=None):
    """Encodes a pandas column with continous ids. 
    """
    # * finds the unique entries of the columns 
    if train_col is not None:
        uniq = train_col.unique()
    else:
        uniq = col.unique()
    # * creates an dictionairy of the unique values with there new value in the column
    name2idx = {o:i for i,o in enumerate(uniq)}
    # * returns the dictonary above, an array with with the new values and -1 if the value was not in the training setm and the numer of unique values  
    return name2idx, np.array([name2idx.get(x, -1) for x in col]), len(uniq)

def encode_data(df, train=None):
    """ Encodes rating data with continous user and movie ids. 
    If train is provided, encodes df with the same encoding as train.
    """
    df = df.copy()

    for col_name in ["user", "item"]:
        train_col = None
        if train is not None:
            train_col = train[col_name]
        _,col,_ = proc_col(df[col_name], train_col)
        df[col_name] = col
        df = df[df[col_name] >= 0]
    return df


class MF(nn.Module):
    # * This function is providing attributes to your class 
    # * the obejct (self) it creates can take in all the attributes below
    def __init__(self, num_users, num_items, emb_size=100, seed=23):
        # * super() is used to create a subclasss 
        super(MF, self).__init__()
        torch.manual_seed(seed)
        # * nn.embeddings = a way to represent categorical or discrete data as continous vectors 
        # * nn.embeddings(input 1, input 2): 
            # * input 1 = amount of input features 
            # * input 2 = amount of output features (the size of the output vectors)
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, emb_size)
        self.item_bias = nn.Embedding(num_items, 1)
        # init 
        self.user_emb.weight.data.uniform_(0,0.05)
        self.item_emb.weight.data.uniform_(0,0.05)
        self.user_bias.weight.data.uniform_(-0.01,0.01)
        self.item_bias.weight.data.uniform_(-0.01,0.01)
# ! so above it looks like this class is creating a object with embeddings for users, user bias, item, item bias 
    
    # * forward = defines the behavior of an object when it is called
    # * forward method is called when the object is used as a function
    # * with the arguments being passed to the function beong passes to the forward method
    # * the forward method is used to define the forward pass of the network, which calculates the output of the network given its inputs.
    # * So if you define def forward() in a PyTorch model class, you are defining the specific behavior of the network when it is called with input data.

# ? ok so i think u will be for users and v will be for items
# * u = torch.LongTensor(df.user.values)
# * v = torch.LongTensor(df.item.values)

    def forward(self, u, v):
        # ### BEGIN SOLUTION
        # u = self.user_emb(u)
        U = self.user_emb(u)
        V = self.item_emb(v)
        b_u = self.user_bias(u).squeeze()
        b_v = self.item_bias(v).squeeze()
        # ? do i have to add that its multiplied by the standard deviation, standard deviation of what? 
        return torch.sigmoid((U*V).sum(1) +  b_u  + b_v)
        ### END SOLUTION
    
def train_one_epoch(model, train_df, optimizer):
    """ Trains the model for one epoch"""
    model.train()
    ### BEGIN SOLUTION
    # * long ternsor means expecting it to be an integer 
    users = torch.LongTensor(train_df.user.values)  #.cuda() 
    items = torch.LongTensor(train_df.items.values) #.cuda()
    # * expecting a float value 
    ratings = torch.FloatTensor(train_df.rating.values)  #.cuda()
    y_hat = model(users, items)
    loss = F.mse_loss(y_hat, ratings)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_loss = loss.items()
    # testloss = valid_loss(model)
  

    ### END SOLUTION
    return train_loss

def valid_metrics(model, valid_df):
    """Computes validation loss and accuracy"""
    model.eval()
    users = torch.LongTensor(valid_df.user.values) # .cuda()
    items = torch.LongTensor(valid_df.item.values) #.cuda()
    ratings = torch.FloatTensor(valid_df.rating.values) #.cuda()
    y_hat = model(users, items)
    valid_loss = F.binary_cross_entropy(y_hat, ratings)
    y_hat1 = y_hat.detach().numpy()
    # valid_loss = vl.detach().numpy()
    # use lose entronep loss binsry loss entropy
    correct = np.sum(ratings == y_hat1)
    total = len(y_hat1)
    valid_acc = correct / total
    ### END SOLUTION
    return valid_loss, valid_acc


def training(model, train_df, valid_df, epochs=10, lr=0.01, wd=0.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for i in range(epochs):
        train_loss = train_one_epoch(model, train_df, optimizer)
        valid_loss, valid_acc = valid_metrics(model, valid_df) 
        print("train loss %.3f valid loss %.3f valid acc %.3f" % (train_loss, valid_loss, valid_acc)) 


#Import some necessary libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#Class importation
from ConcatNet import ConcatNet
from CollabDataset import CollabDataset

#Variables defenition
#Dataset link : https://grouplens.org/datasets/movielens/100k/
data_path = 'ml-100k/'
id_val = 1
num_users = 943 
num_items = 1682

#Data preprocessing
# Dataframe columns : user_id, item_id, rating, and ts.
train_dataframe = pd.read_csv(f'{data_path}u{id_val}.base',sep='\t',header=None)
train_dataframe.columns = ['user_id','item_id','rating','ts']
train_dataframe['user_id'] = train_dataframe['user_id'] -1 
train_dataframe['item_id'] = train_dataframe['item_id'] -1

valid_df = pd.read_csv(f'{data_path}u{id_val}.test',sep='\t',header=None)
valid_df.columns = ['user_id','item_id','rating','ts']
valid_df['user_id'] = valid_df['user_id'] -1 
valid_df['item_id'] = valid_df['item_id'] -1
#Show shapes of data
print('Train data shape : ')
print(train_dataframe.shape)
print('Validation data shape : ')
print(valid_df.shape)

#Check unique users numbers
train_usrs = train_dataframe.user_id.unique()
vald_usrs = valid_df.user_id.unique()
print(len(train_usrs))
print(len(vald_usrs))

train_itms = train_dataframe.item_id.unique()
vald_itms = valid_df.item_id.unique()
print(len(train_itms))
print(len(vald_itms))


#Initiate a CollabDataset class : a dataloader class in pytorch to create batches of the training and validation sets.
#It will return tuples of (user, item, rating).
batch_size = 2000
train_dataset = CollabDataset(train_dataframe)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
valid_dataset = CollabDataset(valid_df)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

print(len(train_dataset))
print(len(valid_dataset))
print(len(train_dataloader))
print(len(valid_dataloader))
print(train_dataset[:3])

#next(iter(train_dataloader))

#ConcaNet model : the recommendation model.
config = {
    'num_users':943, 
    'num_items':1682, 
    'emb_size':50, 
    'emb_droupout': 0.05, 
    'fc_layer_sizes': [100, 512, 256], 
    'dropout': [0.7,0.35], 
    'out_range': [0.8,5.2]} 
model = ConcatNet(config)
#show the architecture
print(model)

#NN parameters
batch_size = 2000 
learning_rate = 1e-2 
weight_decay = 5e-1 
num_epoch = 100 
reduce_learning_rate = 1 
early_stoping = 5 
learning_rates = []
train_losses=[]
valid_losses = []
best_loss = np.inf
best_weights = None

#Data preparation
train_dataset = CollabDataset(train_dataframe)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
valid_dataset = CollabDataset(valid_df)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

#Model initialisation
model = ConcatNet(config)
criterion = nn.MSELoss(reduction='sum')
optim = torch.optim.Adam(model.parameters(), learning_rate=learning_rate, betas=(0.9,0.999), weight_decay=weight_decay)
scheduler = torch.optim.learning_rate_scheduler.Reducelearning_rateOnPlateau(optim, mode='min',factor=0.5, threshold=1e-3,
                                                       patience=reduce_learning_rate, min_learning_rate=learning_rate/10)

for e in range(num_epoch): 
    model.train()
    train_loss = 0
    for u,i,r in train_dataloader:
        r_pred = model(u,i)
        r = r[:,None]
        
        loss = criterion(r_pred,r) 
        optim.zero_grad()
        loss.backward()
        optim.step()
        train_loss+= loss.detach().item()
    current_learning_rate = scheduler.optimizer.param_groudropout[0]['learning_rate']
    learning_rates.append(current_learning_rate)
    train_loss /= len(train_dataset)
    train_losses.append(train_loss)
    
    model.eval()
    valid_loss = 0
    for u,i,r in valid_dataloader:
        r_pred = model(u,i)
        r = r[:,None]
        loss = criterion(r_pred,r)
        valid_loss+=loss.detach().item()
    valid_loss/=len(valid_dataset)
    #record
    valid_losses.append(valid_loss)
    print(f'Epoch {e} Train loss: {train_loss}; Valid loss: {valid_loss}; Learning rate: {current_learning_rate}')

    if valid_loss < best_loss:
        best_loss = valid_loss
        best_weights = deepcopy(model.state_dict())
        no_improvements = 0
    else:
        no_improvements += 1

    if no_improvements >= early_stoping:
        print(f'early stopping after epoch {e}')
        break
    
    scheduler.step(valid_loss)


#Plot the model training curve 
plt.plot(train_losses)
plt.plot(valid_losses)


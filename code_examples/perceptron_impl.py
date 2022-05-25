from cmath import e
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from simple_perceptron import Perceptron

def get_data(directory='./data_samples/perceptron_data.csv'):
      try:
            data = pd.read_csv(directory)
      except FileExistsError as e:
            print('file was not found')
      return data

def slice_data(data):
      X = data.iloc[:, :2]
      y = data.iloc[:, -1]
      return X, y

def train_test_split(X, y):
    shuffle_idx = np.arange(y.shape[0])
    shuffle_rng = np.random.RandomState(123)
    shuffle_rng.shuffle(shuffle_idx)
    X =  X.iloc[shuffle_idx]
    y =  y.iloc[shuffle_idx]

    X_train, X_test = X.iloc[:70], X.iloc[70:]
    y_train, y_test = y.iloc[:70], y.iloc[70:]
    
    return X_train, X_test, y_train, y_test

def standart_normalize(X):
    mu, sigma = X.mean(axis=0), X.std(axis=0)
    return (X - mu) / sigma

def preprocess_data(X, y, device):
      return torch.tensor(X, dtype=torch.float32, device=device), torch.tensor(y, dtype=torch.float32, device=device)

def plot_data(X, y, label='training set'):
      plt.figure(figsize=(12,6))
      plt.scatter(X.X1, X.X2, c=y , marker='s')
      plt.title(label)
      plt.xlabel('X1')
      plt.ylabel('X2')
      plt.xlim([-3, 3]);
      plt.ylim([-3, 3]); 



if __name__ == '__main__':
      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

      data = get_data()
      X, y = slice_data(data)
      X_train, X_test, y_train, y_test = train_test_split(X, y)
      X_train_N = standart_normalize(X_train)
      X_test_N  = standart_normalize(X_test)
      
      ppn = Perceptron(num_features=2, device=device)
      X_train_tensor, y_train_tensor = preprocess_data(X_train_N, y_train, device)

      ppn.train(X_train_tensor, y_train_tensor, epochs=5)
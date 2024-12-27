import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Literal
from torch.utils.data import Dataset


class NasdaqDataset(Dataset):
    def __init__(self, seq_len, mode=Literal["train", "val", "test"], normalize=True):
        self.seq_len = seq_len
        self.normalize = normalize
        self.val_length = 2730
        self.test_length = 2730
        data = pd.read_csv("data/nasdaq100_padding.csv")
        self._preprocess(data, mode)
        
    def _normalize(self, data):
        train = data[:-self.val_length-self.test_length]
        scaler = MinMaxScaler()
        scaler.fit(train)
        data_normalized = scaler.transform(data)
        return scaler, data_normalized 
        
    def _preprocess(self, data, mode):
        X = data.drop(columns=["NDX"]).values       # (N, T, F)
        y = data["NDX"].values.reshape(-1, 1)       # (N, 1)
        target = data["NDX"].values.reshape(-1, 1)  # (N, 1)
        
        if self.normalize:
            self.scaler_x, X = self._normalize(X)
            self.scaler_y, y = self._normalize(y)
            self.scaler_target, target = self._normalize(target)
        X_sliced, y_sliced, target = self._slice_data(X, y, target)
        
        if mode == "train":
            self.X = torch.from_numpy(X_sliced[:-self.val_length-self.test_length]).float()
            self.y = torch.from_numpy(y_sliced[:-self.val_length-self.test_length]).float()
            self.target = torch.from_numpy(target[:-self.val_length-self.test_length]).float()
        elif mode == "val":
            self.X = torch.from_numpy(X_sliced[-self.val_length-self.test_length:-self.test_length]).float()
            self.y = torch.from_numpy(y_sliced[-self.val_length-self.test_length:-self.test_length]).float()
            self.target = torch.from_numpy(target[-self.val_length-self.test_length:-self.test_length]).float()
        else:
            self.X = torch.from_numpy(X_sliced[-self.test_length:]).float()
            self.y = torch.from_numpy(y_sliced[-self.test_length:]).float()
            self.target = torch.from_numpy(target[-self.test_length:]).float()
        
    def _slice_data(self, X, y, target):
        counts = len(X) - self.seq_len + 1  
        X_sliced = np.zeros((counts, self.seq_len, X.shape[1])) #(N-T+1, T, F)
        for i in range(counts):
            X_sliced[i] = X[i:i+self.seq_len]
            
        y_sliced = np.zeros((counts, self.seq_len-1, 1)) #(N-T+1, T-1, 1)
        for i in range(counts):
            y_sliced[i] = y[i:i+self.seq_len-1]

        target = target[self.seq_len-1:, :] #(N-T+1, 1)
        return X_sliced, y_sliced, target
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.target[idx]


class NasdaqDataset_Exp2(Dataset):
    def __init__(self, seq_len, mode=Literal["train", "val", "test"], normalize=True):
        self.seq_len = seq_len
        self.normalize = normalize
        self.val_length = 2730
        self.test_length = 2730
        data = pd.read_csv("data/nasdaq100_padding.csv")
        self._preprocess(data, mode)
        
    def _normalize(self, data):
        train = data[:-self.val_length-self.test_length]
        scaler = MinMaxScaler()
        scaler.fit(train)
        data_normalized = scaler.transform(data)
        return scaler, data_normalized 
        
    def _preprocess(self, data, mode):
        X = data.drop(columns=["NDX"]).values       # (N, T, F)
        
        # add noise
        X = np.concat([X, np.random.randn(X.shape[0], X.shape[1])], axis=1)
        
        y = data["NDX"].values.reshape(-1, 1)       # (N, 1)
        target = data["NDX"].values.reshape(-1, 1)  # (N, 1)
        
        if self.normalize:
            self.scaler_x, X = self._normalize(X)
            self.scaler_y, y = self._normalize(y)
            self.scaler_target, target = self._normalize(target)
        X_sliced, y_sliced, target = self._slice_data(X, y, target)
        
        if mode == "train":
            self.X = torch.from_numpy(X_sliced[:-self.val_length-self.test_length]).float()
            self.y = torch.from_numpy(y_sliced[:-self.val_length-self.test_length]).float()
            self.target = torch.from_numpy(target[:-self.val_length-self.test_length]).float()
        elif mode == "val":
            self.X = torch.from_numpy(X_sliced[-self.val_length-self.test_length:-self.test_length]).float()
            self.y = torch.from_numpy(y_sliced[-self.val_length-self.test_length:-self.test_length]).float()
            self.target = torch.from_numpy(target[-self.val_length-self.test_length:-self.test_length]).float()
        else:
            self.X = torch.from_numpy(X_sliced[-self.test_length:]).float()
            self.y = torch.from_numpy(y_sliced[-self.test_length:]).float()
            self.target = torch.from_numpy(target[-self.test_length:]).float()
        
    def _slice_data(self, X, y, target):
        counts = len(X) - self.seq_len + 1  
        X_sliced = np.zeros((counts, self.seq_len, X.shape[1])) #(N-T+1, T, F)
        for i in range(counts):
            X_sliced[i] = X[i:i+self.seq_len]
            
        y_sliced = np.zeros((counts, self.seq_len-1, 1)) #(N-T+1, T-1, 1)
        for i in range(counts):
            y_sliced[i] = y[i:i+self.seq_len-1]

        target = target[self.seq_len-1:, :] #(N-T+1, 1)
        return X_sliced, y_sliced, target
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.target[idx]



class NasdaqDataset2(Dataset):
    def __init__(self, seq_len, mode=Literal["train", "val", "test"], normalize=True):
        self.seq_len = seq_len
        self.normalize = normalize
        self.val_length = 2730
        self.test_length = 2730
        data = pd.read_csv("data/nasdaq100_padding.csv")
        
        self._preprocess(data, mode)
        
    def _normalize_for_X(self, data):
        epsilon = 1e-8
        mean = data.mean(axis=1, keepdim=True)
        std = data.std(axis=1, keepdim=True)
        normalized_data = (data-mean)/(std+epsilon)
        
        return normalized_data, mean, std
    
    def _normalize_for_y(self, data):
        epsilon = 1e-8
        past_data= data[:,:-1,:]
        mean = past_data.mean(axis=1, keepdim=True)
        std = past_data.std(axis=1, keepdim=True)
        normalized_data = (data-mean)/(std+epsilon)
        
        return normalized_data, mean, std
    
    def _extract_sliding_windows(self, raw_data, window):
        raw_data = torch.from_numpy(raw_data.values).float()
        sample_n = len(raw_data) - window + 1
        n_feature = raw_data.shape[-1]
        data = torch.zeros(sample_n, window, n_feature)
        for i in range(sample_n):
            start = i
            end = i + window    
            data[i, :, :] = raw_data[start:end]
        return data

    def _preprocess(self, data, mode):
        data = self._extract_sliding_windows(data, self.seq_len)
        X = data[:, :, :-1] # (N, T, F)
        y = data[:, :, -1:] # (N, T, 1)

        norm_X, mean_X, std_X = self._normalize_for_X(X)
        # norm_y, mean_y, std_y = self._normalize_for_y(y)
        norm_y, mean_y, std_y = self._normalize_for_X(y)
        norm_y = norm_y[:, :-1, :] # (N, T-1, 1)
        norm_target = norm_y[:, -1:, :] # (N, 1, 1)
        
        if mode == "train":
            self.X = norm_X[:-self.val_length-self.test_length]
            self.y = norm_y[:-self.val_length-self.test_length]
            self.target = norm_target[:-self.val_length-self.test_length]
            self.mean_X = mean_X[:-self.val_length-self.test_length]
            self.std_X = std_X[:-self.val_length-self.test_length]
            self.mean_y = mean_y[:-self.val_length-self.test_length]
            self.std_y = std_y[:-self.val_length-self.test_length]
        elif mode == "val":
            self.X = norm_X[-self.val_length-self.test_length:-self.test_length]
            self.y = norm_y[-self.val_length-self.test_length:-self.test_length]
            self.target = norm_target[-self.val_length-self.test_length:-self.test_length]
            self.mean_X = mean_X[-self.val_length-self.test_length:-self.test_length]
            self.std_X = std_X[-self.val_length-self.test_length:-self.test_length]
            self.mean_y = mean_y[-self.val_length-self.test_length:-self.test_length]
            self.std_y = std_y[-self.val_length-self.test_length:-self.test_length]
        else:
            self.X = norm_X[-self.test_length:]
            self.y = norm_y[-self.test_length:]
            self.target = norm_target[-self.test_length:]
            self.mean_X = mean_X[-self.test_length:]
            self.std_X = std_X[-self.test_length:]
            self.mean_y = mean_y[-self.test_length:]
            self.std_y = std_y[-self.test_length:]
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.target[idx]



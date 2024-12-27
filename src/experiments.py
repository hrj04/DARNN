import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm

import torch
from torch import nn
from warnings import filterwarnings
from src.data import NasdaqDataset, NasdaqDataset_Exp2
from torch.utils.data import DataLoader
from src.models import DARNN, DARNN_Exp2
import sys 

filterwarnings("ignore")

def Base_Table2():
    data = pd.read_csv("data/nasdaq100_padding.csv")
    target = data["NDX"]
    
    # Split data into train, validation, and test sets
    train_length = 35100
    val_length = 2730
    test_length = 2730

    target_test = target[-test_length:].values
    
    # base score
    # just using past one-step data as predictor
    pred_base = target_test[:-1]
    true_base = target_test[1:]

    mae = mean_absolute_error(true_base, pred_base)
    mse = mean_squared_error(true_base, pred_base)
    mape = mean_absolute_percentage_error(true_base, pred_base)

    print("Test Set Metrics:")
    print(f"RMSE : {np.sqrt(mse):0.2f}, MAE : {mae:0.2f}, MAPE : {mape*10000:0.2f}(x10^-2%)")
    
    return pred_base, true_base
    
def ARIMA_Tabel2():
    data = pd.read_csv("data/nasdaq100_padding.csv")
    target = data["NDX"]
    
    # Split data into train, validation, and test sets
    train_length = 35100
    val_length = 2730
    test_length = 2730

    target_train = target[:train_length+val_length].values
    target_test = target[-test_length:].values
    
    preds = []
    trues = []
    # Fit the ARIMA model on the training set
    history = target_train  # Initial history
    order = (10, 1, 0)  # ARIMA(hidden_dim_decoder, d, q) order
    model = ARIMA(history, order=order)
    model_fit = model.fit()

    # Forecast one step ahead
    pred = model_fit.forecast(steps=1)
    preds.append(pred[0])
    trues.append(target_test[0])

    for step in tqdm(range(1, len(target_test))):
        # Update the model with the latest observation
        model_fit = model_fit.append(target_test[step-1:step])
        
        # Forecast one step ahead
        pred = model_fit.forecast(steps=1)
        preds.append(pred[0])
        trues.append(target_test[step])

    preds = np.array(preds)
    trues = np.array(trues)
    target_test.shape
    mse_test = mean_squared_error(preds, trues)
    mae_test = mean_absolute_error(preds, trues)
    mape_test = mean_absolute_percentage_error(preds, trues)

    print("Test Set Metrics:")
    print(f"RMSE : {np.sqrt(mse_test):0.2f}, MAE : {mae_test:0.2f}, MAPE : {mape_test*10000:0.2f}(x10^-2%)")
    
    return preds, trues    

def DARNN_Table2(seq_len, hidden_dim_encoder, hidden_dim_decoder, batch_size, epochs, seed, load_model=False):
    weight_path = f"weights/DARNN_L{seq_len}_HDE{hidden_dim_encoder}_HDD{hidden_dim_decoder}_BS{batch_size}_E{epochs}_s{seed}.pth"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # load data
    train_dataset = NasdaqDataset(seq_len=seq_len, mode='train', normalize=True)
    val_dataset = NasdaqDataset(seq_len=seq_len, mode='val', normalize=True)
    test_dataset = NasdaqDataset(seq_len=seq_len, mode='test', normalize=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    stock_num = train_dataset.X.shape[2]
    seq_len = train_dataset.X.shape[1]

    # define model
    model = DARNN(hidden_dim_decoder=hidden_dim_decoder, 
                  hidden_dim_encoder=hidden_dim_encoder, 
                  stock_num=stock_num, 
                  seq_len=seq_len
                  ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = nn.MSELoss()
    
    if load_model:
        model.load_state_dict(torch.load(weight_path))
        model.eval()
        preds = []
        true = []
        for batch_x, batch_y, batch_target in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_target = batch_target.to(device)
            output = model(batch_x, batch_y) # (batch, 1)
            preds.append(output.detach().cpu().numpy())
            true.append(batch_target.detach().cpu().numpy())
        preds = np.concatenate(preds)
        true = np.concatenate(true)
        
        # inverse transform
        preds = test_dataset.scaler_target.inverse_transform(preds)
        true = test_dataset.scaler_target.inverse_transform(true)
        
        # calculate metrics
        rmse = np.sqrt(mean_squared_error(true, preds))
        mae = mean_absolute_error(true, preds)
        mape = mean_absolute_percentage_error(true, preds)
        
        return rmse, mae, mape
    
    else:
        best_test_rmse = 1e10
        best_test_mae = 1e10
        best_test_mape = 1e10
        pbar = tqdm(range(epochs))
        for e in pbar:
            mse_train = 0
            model.train()
            for batch_x, batch_y, batch_target in train_loader:
                batch_x = batch_x.to(device) # (batch, seq_len, N)
                batch_y = batch_y.to(device) # (batch, seq_len-1, 1)
                batch_target = batch_target.to(device) # (batch, 1)
                opt.zero_grad()
                y_pred = model(batch_x, batch_y) # (batch, 1)
                l = loss(y_pred, batch_target)
                l.backward()
                
                # Gradient Clipping to stabilize training
                max_norm = 1 
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                mse_train += l.item()
                opt.step()
            train_loss = mse_train / len(train_loader)

            mse_val = 0
            model.eval()
            for batch_x, batch_y, batch_target in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_target = batch_target.to(device)
                output = model(batch_x, batch_y) # (batch, 1)
                mse_val += loss(output, batch_target).item()
            val_loss = mse_val / len(val_loader)

            # realtime test eval for experiments
            preds = []
            true = []
            for batch_x, batch_y, batch_target in test_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_target = batch_target.to(device)
                output = model(batch_x, batch_y) # (batch, 1)
                preds.append(output.detach().cpu().numpy())
                true.append(batch_target.detach().cpu().numpy())
            preds = np.concatenate(preds)
            true = np.concatenate(true)
            
            # inverse transform
            preds = test_dataset.scaler_target.inverse_transform(preds)
            true = test_dataset.scaler_target.inverse_transform(true)
            
            # calculate metrics
            rmse = np.sqrt(mean_squared_error(true, preds))
            mae = mean_absolute_error(true, preds)
            mape = mean_absolute_percentage_error(true, preds)
            
            if best_test_rmse > rmse:
                best_test_rmse = rmse
                best_test_mae = mae
                best_test_mape = mape
                torch.save(model.state_dict(), weight_path)
                print("\n")
                print("########################################################################")
                print(f"L{seq_len}_HDE{hidden_dim_encoder}_HDD{hidden_dim_decoder}_BS{batch_size}_E{epochs}_s{seed}")
                print(f"Best RMSE : {rmse:.4f}, MAE : {mae:.4f}, MAPE : {mape*10000:.4f}10^-2%")
                print("########################################################################")
                sys.stdout.flush()
            pbar.set_postfix({'train_loss': round(train_loss,5), 'val_loss': round(val_loss,5)})
        return best_test_rmse, best_test_mae, best_test_mape
    
def DARNN_Figure3(seq_len, hidden_dim_encoder, hidden_dim_decoder, batch_size, epochs, seed, load_model=False):
    weight_path = f"weights/DARNN_Figure3_L{seq_len}_HDE{hidden_dim_encoder}_HDD{hidden_dim_decoder}_BS{batch_size}_E{epochs}_s{seed}.pth"
    alpha_path = f"weights/DARNN_Figure3_L{seq_len}_HDE{hidden_dim_encoder}_HDD{hidden_dim_decoder}_BS{batch_size}_E{epochs}_s{seed}"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # load data
    train_dataset = NasdaqDataset_Exp2(seq_len=seq_len, mode='train', normalize=True)
    val_dataset = NasdaqDataset_Exp2(seq_len=seq_len, mode='val', normalize=True)
    test_dataset = NasdaqDataset_Exp2(seq_len=seq_len, mode='test', normalize=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    stock_num = train_dataset.X.shape[2]
    seq_len = train_dataset.X.shape[1]

    # define model
    model = DARNN_Exp2(hidden_dim_decoder=hidden_dim_decoder, 
                       hidden_dim_encoder=hidden_dim_encoder,
                       stock_num=stock_num,
                       seq_len=seq_len
                       ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = nn.MSELoss()
    
    best_test_rmse = 1e10
    best_test_mae = 1e10
    best_test_mape = 1e10
    pbar = tqdm(range(epochs))
    for e in pbar:
        mse_train = 0
        model.train()
        for batch_x, batch_y, batch_target in train_loader:
            batch_x = batch_x.to(device) # (batch, seq_len, N)
            batch_y = batch_y.to(device) # (batch, seq_len-1, 1)
            batch_target = batch_target.to(device) # (batch, 1)
            opt.zero_grad()
            y_pred, train_alpha_t_k = model(batch_x, batch_y) # (batch, 1)
            l = loss(y_pred, batch_target)
            l.backward()
            
            # Gradient Clipping to stabilize training
            max_norm = 1 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            mse_train += l.item()
            opt.step()
        train_loss = mse_train / len(train_loader)
        np.save(f"{alpha_path}_train_alpha.npy", train_alpha_t_k[0,0,:].cpu().detach().numpy())
        
        mse_val = 0
        model.eval()
        for batch_x, batch_y, batch_target in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_target = batch_target.to(device)
            output, _ = model(batch_x, batch_y) # (batch, 1)
            mse_val += loss(output, batch_target).item()
        val_loss = mse_val / len(val_loader)
        
        # realtime test eval for experiments
        preds = []
        true = []
        for batch_x, batch_y, batch_target in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_target = batch_target.to(device)
            output, test_alpha_t_k = model(batch_x, batch_y) # (batch, 1)
            preds.append(output.detach().cpu().numpy())
            true.append(batch_target.detach().cpu().numpy())
        preds = np.concatenate(preds)
        true = np.concatenate(true)
        
        # inverse transform
        preds = test_dataset.scaler_target.inverse_transform(preds)
        true = test_dataset.scaler_target.inverse_transform(true)
        
        # calculate metrics
        rmse = np.sqrt(mean_squared_error(true, preds))
        mae = mean_absolute_error(true, preds)
        mape = mean_absolute_percentage_error(true, preds)
        
        if best_test_rmse > rmse:
            best_test_rmse = rmse
            best_test_mae = mae
            best_test_mape = mape
            test_alphas = test_alpha_t_k
            torch.save(model.state_dict(), weight_path)
            np.save(f"{alpha_path}_test_alpha.npy", test_alphas[0,0,:].cpu().detach().numpy())
            print("\n")
            print("########################################################################")
            print(f"L{seq_len}_HDE{hidden_dim_encoder}_HDD{hidden_dim_decoder}_BS{batch_size}_E{epochs}_s{seed}")
            print(f"Best RMSE : {rmse:.4f}, MAE : {mae:.4f}, MAPE : {mape*10000:.4f}10^-2%")
            print("########################################################################")
            sys.stdout.flush()
        pbar.set_postfix({'train_loss': round(train_loss,5), 'val_loss': round(val_loss,5)})
    return best_test_rmse, best_test_mae, best_test_mape
    

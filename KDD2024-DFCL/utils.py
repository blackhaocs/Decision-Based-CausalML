import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class CriteoDataset(Dataset):
    def __init__(self, X, T, Y_visit, Y_visit_float, Y_conv, Y_conv_float):
        super(CriteoDataset, self).__init__()
        self.X = X.astype(np.float32)
        self.T = T.astype(np.float32)
        self.Y_visit = Y_visit.astype(np.float32)
        self.Y_conv = Y_conv.astype(np.float32)
        self.Y_visit_float = Y_visit_float.astype(np.float32)
        self.Y_conv_float = Y_conv_float.astype(np.float32)

    def __getitem__(self, index):
        return self.X[index], self.T[index], self.Y_visit[index], self.Y_visit_float[index], self.Y_conv[index], self.Y_conv_float[index]

    def __len__(self):
        return len(self.X)


def get_dataloader(X, T, Y_visit, Y_visit_float, Y_conv, Y_conv_float, batch_size=10000, num_workers=4):
    dataset = CriteoDataset(X=X, T=T, Y_visit=Y_visit, Y_visit_float=Y_visit_float, Y_conv=Y_conv, Y_conv_float=Y_conv_float)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    return dataloader


def scaling(x, min, max):
    return np.where(x < min, 0.0, np.where(x > max, 1.0, (x - min) / (max - min)))

def get_data(df, weights, unique, BATCH_SIZE=32768, num_workers=4):
    X = df[['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11']].values.astype(np.float32)
    X[:, 0] = scaling(X[:, 0], min=np.min(X[:, 0]), max=np.max(X[:, 0]))
    X[:, 1] = scaling(X[:, 1], min=np.min(X[:, 1]), max=np.max(X[:, 1]))
    X[:, 2] = scaling(X[:, 2], min=np.min(X[:, 2]), max=np.max(X[:, 2]))
    X[:, 3] = scaling(X[:, 3], min=np.min(X[:, 3]), max=np.max(X[:, 3]))
    X[:, 4] = scaling(X[:, 4], min=np.min(X[:, 4]), max=np.max(X[:, 4]))
    X[:, 5] = scaling(X[:, 5], min=np.min(X[:, 5]), max=np.max(X[:, 5]))
    X[:, 6] = scaling(X[:, 6], min=np.min(X[:, 6]), max=np.max(X[:, 6]))
    X[:, 7] = scaling(X[:, 7], min=np.min(X[:, 7]), max=np.max(X[:, 7]))
    X[:, 8] = scaling(X[:, 8], min=np.min(X[:, 8]), max=np.max(X[:, 8]))
    X[:, 9] = scaling(X[:, 9], min=np.min(X[:, 9]), max=np.max(X[:, 9]))
    X[:, 10] = scaling(X[:, 10], min=np.min(X[:, 10]), max=np.max(X[:, 10]))
    X[:, 11] = scaling(X[:, 11], min=np.min(X[:, 11]), max=np.max(X[:, 11]))

    T = df['treatment'].values.reshape(-1, 1)
    Y_visit = df['visit'].values.reshape(-1, 1)
    Y_visit_float = df['visit'].values.reshape(-1, 1).astype(np.float32)
    Y_conv = df['conversion'].values.reshape(-1, 1)
    Y_conv_float = df['conversion'].values.reshape(-1, 1).astype(np.float32)
    
    for i in unique:
        idx = np.where(T == i)[0]
        Y_conv_float[idx] = Y_conv_float[idx] / weights[i]
        Y_visit_float[idx] = Y_visit_float[idx] / weights[i]
                
    train_len = int(len(X) * 0.6)
    val_len = int(len(X) * 0.7)
    

    X_train = X[:train_len, :]
    T_train = T[:train_len, :]
    Y_visit_train = Y_visit[:train_len, :]
    Y_conv_train = Y_conv[:train_len, :]
    Y_visit_float_train = Y_visit_float[:train_len, :]
    Y_conv_float_train = Y_conv_float[:train_len, :]

    X_val = X[train_len:val_len, :]
    T_val = T[train_len:val_len, :]
    Y_visit_val = Y_visit[train_len:val_len, :]
    Y_conv_val = Y_conv[train_len:val_len, :]
    Y_visit_float_val = Y_visit_float[train_len:val_len, :]
    Y_conv_float_val = Y_conv_float[train_len:val_len, :]

    X_test = X[val_len:, :]
    T_test = T[val_len:, :]
    Y_visit_test = Y_visit[val_len:, :]
    Y_conv_test = Y_conv[val_len:, :]
    Y_visit_float_test = Y_visit_float[val_len:, :]
    Y_conv_float_test = Y_conv_float[val_len:, :]
    
    X_val = torch.from_numpy(X_val).to(torch.float32)
    T_val = torch.from_numpy(T_val).to(torch.float32)
    Y_visit_val = torch.from_numpy(Y_visit_val).to(torch.float32)
    Y_conv_val = torch.from_numpy(Y_conv_val).to(torch.float32)
    Y_visit_float_val = torch.from_numpy(Y_visit_float_val).to(torch.float32)
    Y_conv_float_val = torch.from_numpy(Y_conv_float_val).to(torch.float32)

    X_test = torch.from_numpy(X_test).to(torch.float32)
    T_test = torch.from_numpy(T_test).to(torch.float32)
    Y_visit_test = torch.from_numpy(Y_visit_test).to(torch.float32)
    Y_conv_test = torch.from_numpy(Y_conv_test).to(torch.float32)
    Y_visit_float_test = torch.from_numpy(Y_visit_float_test).to(torch.float32)
    Y_conv_float_test = torch.from_numpy(Y_conv_float_test).to(torch.float32)
    
    dl = get_dataloader(X=X_train, T=T_train, Y_visit=Y_visit_train, Y_visit_float=Y_visit_float_train, Y_conv=Y_conv_train, Y_conv_float=Y_conv_float_train, batch_size=BATCH_SIZE, num_workers=num_workers)

    return dl, X_val, T_val, Y_visit_val, Y_visit_float_val, Y_conv_val, Y_conv_float_val, X_test, T_test, Y_visit_test, Y_visit_float_test, Y_conv_test, Y_conv_float_test


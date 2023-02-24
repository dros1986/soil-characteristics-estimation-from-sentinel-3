import os
import math
import random
import torch
import pandas as pd
import numpy as np
from scipy import signal
from sklearn.exceptions import NotFittedError

# ['coarse','clay','silt','sand','pH.in.CaCl2','pH.in.H2O','OC','CaCO3','N','P','K','CEC']


class DatasetLucas(torch.utils.data.Dataset):
    def __init__(self,
            conf,
            csv_key,
            src_norm,
            tgt_norm,
            binner = None,
            return_coords = False,
            fillna = True,
        ):
        # save attributes
        self.conf = conf
        self.src_norm = src_norm
        self.tgt_norm = tgt_norm
        # read csv files
        df = pd.read_table(self.conf[csv_key], sep=self.conf['sep'])
        # fill for missing files
        if fillna: df = df.fillna(0)
        # get source and target variables
        self.src_vars = self.get_vars(df, self.conf['src_vars'])
        self.tgt_vars = self.get_vars(df, self.conf['tgt_vars'])
        # check size
        assert(self.src_vars.shape[0] == self.tgt_vars.shape[0])
        # save min and max of target data
        self.tgt_min = self.tgt_vars.min(0)[0]
        self.tgt_avg = self.tgt_vars.mean(0)
        self.tgt_max = self.tgt_vars.max(0)[0]
        # normalize input features
        self.src_vars = self.src_norm(self.src_vars)
        # quantize in bins if required
        if binner is not None:
            # check if binner has been already trained
            try:
                # if so, create bins
                cur_encoding = binner.transform(self.tgt_vars)
            except NotFittedError as e:
                # else train and create bins
                cur_encoding = binner.fit_transform(self.tgt_vars) #.float() # .toarray()
            # set classes
            self.tgt_vars = torch.from_numpy(cur_encoding).float()
        else:
            # normalize also target variables
            self.tgt_vars = self.tgt_norm(self.tgt_vars)
        # get coordinates
        self.return_coords = return_coords
        self.coords = torch.from_numpy(df[[self.conf['gps_lat'], self.conf['gps_lon']]].to_numpy()).float()
        
    
    def get_num_src_vars(self):
        return self.src_vars.shape[1]
    
    def get_num_tgt_vars(self):
        return self.tgt_vars.shape[1]
    
    def get_min_max_avg(self):
        return self.tgt_min, self.tgt_max, self.tgt_avg

    def get_vars(self, df, tgt_vars):
        ''' gets variables using specified columns '''
        res = []
        # for each column of interest
        for cur_col in tgt_vars:
            # if it's of object type, evaluate it to convert from str to list
            if df[cur_col].dtype == object:
                df[cur_col] = df[cur_col].apply(eval)
            # get current feature
            cur_feat = np.array(df[cur_col].tolist())
            # add second dimension
            if cur_feat.ndim == 1:
                cur_feat = np.expand_dims(cur_feat, axis = 1)
            # append
            res.append(cur_feat)
        # join all mat
        res = np.concatenate(res, axis=1)
        # convert to torch
        res = torch.from_numpy(res).float()
        # return them
        return res


    def __len__(self):
        return self.src_vars.shape[0]


    def __getitem__(self, idx):
        src = self.src_vars[idx].float()
        tgt = self.tgt_vars[idx].float()
        if self.return_coords:
            return src, tgt, self.coords[idx]
        return src, tgt







class DatasetLucasIterator(object):
    def __init__(self,
            conf,
            csv_key,
            src_norm,
            tgt_norm,
            binner = None,
            drop_last = True,
            shuffle = True,
            return_coords = False,
            fillna = True,
        ):
        # save attributes
        self.conf = conf
        self.src_norm = src_norm
        self.tgt_norm = tgt_norm
        self.shuffle = shuffle
        # read csv files
        df = pd.read_table(self.conf[csv_key], sep=self.conf['sep'])
        # fill for missing files
        if fillna: df = df.fillna(0)
        # get source and target variables
        self.src_vars = self.get_vars(df, self.conf['src_vars'])
        self.tgt_vars = self.get_vars(df, self.conf['tgt_vars'])
        # check size
        assert(self.src_vars.shape[0] == self.tgt_vars.shape[0])
        # save min and max of target data
        self.tgt_min = self.tgt_vars.min(0)[0]
        self.tgt_avg = self.tgt_vars.mean(0)
        self.tgt_max = self.tgt_vars.max(0)[0]
        # normalize input features
        self.src_vars = self.src_norm(self.src_vars)
        # quantize in bins if required
        if binner is not None:
            # check if binner has been already trained
            try:
                # if so, create bins
                cur_encoding = binner.transform(self.tgt_vars)
            except NotFittedError as e:
                # else train and create bins
                cur_encoding = binner.fit_transform(self.tgt_vars) #.float() # .toarray()
            # set classes
            self.tgt_vars = torch.from_numpy(cur_encoding).float()
        else:
            # normalize also target variables
            self.tgt_vars = self.tgt_norm(self.tgt_vars)
        # get coordinates
        self.return_coords = return_coords
        self.coords = torch.from_numpy(df[[self.conf['gps_lat'], self.conf['gps_lon']]].to_numpy()).float()
        # define number of batches
        if drop_last:
            self.n_batches = self.tgt_vars.size(0) // self.conf['batch_size']
        else:
            self.n_batches = math.ceil(self.tgt_vars.size(0) / self.conf['batch_size'])
        # define random sequence
        self.seq = list(range(self.tgt_vars.shape[0]))
        if self.shuffle:
            random.shuffle(self.seq)
        # define current batch number
        self.cur_batch = 0
        
    
    def get_num_src_vars(self):
        return self.src_vars.shape[1]
    
    def get_num_tgt_vars(self):
        return self.tgt_vars.shape[1]
    
    def get_min_max_avg(self):
        return self.tgt_min, self.tgt_max, self.tgt_avg

    def get_vars(self, df, tgt_vars):
        ''' gets variables using specified columns '''
        res = []
        # for each column of interest
        for cur_col in tgt_vars:
            # if it's of object type, evaluate it to convert from str to list
            if df[cur_col].dtype == object:
                df[cur_col] = df[cur_col].apply(eval)
            # get current feature
            cur_feat = np.array(df[cur_col].tolist())
            # add second dimension
            if cur_feat.ndim == 1:
                cur_feat = np.expand_dims(cur_feat, axis = 1)
            # append
            res.append(cur_feat)
        # join all mat
        res = np.concatenate(res, axis=1)
        # convert to torch
        res = torch.from_numpy(res).float()
        # return them
        return res


    def __iter__(self):
        return self

    def __next__(self):
        if not self.cur_batch < self.n_batches:
            self.cur_batch = 0
            if self.shuffle:
                random.shuffle(self.seq)
            raise StopIteration
        else:
            isrt = self.cur_batch*self.conf['batch_size']
            iend = isrt + self.conf['batch_size']
            ids = self.seq[isrt:iend]
            self.cur_batch += 1
            src = self.src_vars[ids].float()
            tgt = self.tgt_vars[ids].float()
            if self.return_coords:
                return src, tgt, self.coords[ids]
            return src, tgt

    def __len__(self):
        return self.n_batches


if __name__ == '__main__':

    def test_dataloader(data, epochs=20):
        for cur_epoch in range(epochs):
            # check
            for i, (cur_in, cur_gt) in enumerate(data):
                print(cur_in.shape)
                print(cur_gt.shape)
                print(i)
            print('\n', 50*'#','\n')


    # import only for testing
    from Normalization import VariableStandardization
    from sklearn.preprocessing import KBinsDiscretizer
    
    # instantiate normalizer
    src_norm = VariableStandardization()
    tgt_norm = VariableStandardization()
    # create binner
    binner = None
    # binner = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    # define configuration 
    conf = {
        'val_csv':'/home/flavio/Documenti/pignoletto/dati_chiara/valchiavenna_v2/ml_feats/val.csv',
        'sep':',',
        'src_vars':['lbp_uniform_p8_r2', 'WindExpos'],
        'tgt_vars':['pH1', 'pH12'],
        'gps_lat':'X',
        'gps_lon':'Y',
    }
    # create dataset
    ds = DatasetLucas(conf, 'val_csv', src_norm, tgt_norm, binner=binner)
    loader = torch.utils.data.DataLoader(ds, batch_size=20, shuffle=True, num_workers=0)
    
    test_dataloader(loader, epochs=20)
    
    
    
    
    


import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
from pprint import pprint

from Normalization import *
from DatasetLucas import DatasetLucas
from networks import Net
# from Renderer import Renderer
from test_functions import test_batch_classification, test_batch_regression

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import RichProgressBar


class Experiment(pl.LightningModule):
    def __init__(self, conf):
        super(Experiment, self).__init__()
        # fix params
        conf = self.fix_and_set_default_params(conf)
        # save parameters
        self.save_hyperparameters(conf) # not working
        # self.hparams = conf
        self.conf = conf
        # infer number of variables from test dataset
        # NOTE: slow but only way to get number of variables        
        self.src_norm = NoOp()
        self.tgt_norm = NoOp()
        self.binner = None
        dl = self.test_dataset()
        # define input and output size
        inpsz = dl.get_num_src_vars()
        outsz = dl.get_num_tgt_vars()
        # create normalization objects
        self.src_norm = VariableStandardization(inpsz)
        self.tgt_norm = VariableStandardization(outsz)
        # create binning objects
        self.binner = None
        if conf['nclasses'] > 0:
            self.binner = KBinsDiscretizer(n_bins=conf['nclasses'], encode='ordinal', strategy='quantile')
        # define number of outputs
        nout = outsz*conf['nclasses'] if conf['nclasses'] > 0 else outsz
        # define network
        self.net = Net(self.conf, ninp=inpsz, nout=nout)
        # define metric
        self.loss_fun = self.get_loss(conf['loss'])
        # init min/max/avg
        self.register_buffer('tgt_min', torch.zeros(len(conf['tgt_vars'])))
        self.register_buffer('tgt_max', torch.ones(len(conf['tgt_vars'])))
        self.register_buffer('tgt_avg', torch.ones(len(conf['tgt_vars'])))



    def fix_and_set_default_params(self, conf):
        # if src_vars and tgt_vars are not lists, make them lists with one element
        if not isinstance(conf['src_vars'], list): conf['src_vars'] = [conf['src_vars']]
        if not isinstance(conf['tgt_vars'], list): conf['tgt_vars'] = [conf['tgt_vars']]
        # set number of classes to 0 if not specified (it's a regression then)
        if 'nclasses' not in conf: conf['nclasses'] = 0
        # if classification, set cross entropy as loss
        if conf['nclasses'] > 0: conf['loss'] = 'cross_entropy'
        # return fixed configuration
        return conf


    def forward(self, x):
        return self.net(x)


    def one_hot_to_classes(self, onehot):
        # define number of classes
        nclasses = self.conf['nclasses']
        # define number of target variables
        ntargets = len(self.conf['tgt_vars'])
        # init output
        out = torch.zeros(onehot.shape[0], ntargets)
        # for each target
        for i in range(ntargets):
            # get range
            rng_srt, rng_end = i*nclasses, (i+1)*nclasses
            # 
            out[:,i] = torch.argmax(onehot[:,rng_srt:rng_end],1)
        # return classes
        return out


    def training_step(self, batch, batch_nb):
        # split components
        src, tgt = batch
        # get prediction
        out = self(src)
        # TODO: project in output space
        # apply loss
        loss = self.loss_fun(out, tgt)
        # return loss value
        self.log("train_loss", loss, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        # split components
        src, tgt = batch
        # compute prediction
        with torch.no_grad():
            out = self(src)
        # get classes if classification
        if self.conf['nclasses'] > 0:
            out = self.one_hot_to_classes(out)
        # return output and gt
        return out, tgt
        # # reverse normalization and quantization
        # tgt = self.tgt_norm.invert(tgt) #.cpu())
        # out = self.tgt_norm.invert(out) #.cpu())


    def validation_epoch_end(self, outputs):
        # get outputs and gts
        out, tgt = zip(*outputs)
        # concatenate
        out = torch.cat(out, 0)
        tgt = torch.cat(tgt, 0)
        # invert
        out = self.tgt_norm.invert(out)
        tgt = self.tgt_norm.invert(tgt)
        # compute metrics
        if self.conf['nclasses'] > 0:
            cur_val = test_batch_classification(out, tgt, self.tgt_min, self.tgt_max, self.tgt_avg, self.conf['tgt_vars'])
        else:
            cur_val = test_batch_regression(out, tgt, self.tgt_min, self.tgt_max, self.tgt_avg, self.conf['tgt_vars'])
        # log
        # import ipdb; ipdb.set_trace()
        for cur_key in cur_val:
            self.log(cur_key, cur_val[cur_key], prog_bar=True)
        # self.log_dict(cur_val, prog_bar=True)



    def test_step(self, batch, batch_idx):
        # split components
        src, tgt = batch
        # compute prediction
        with torch.no_grad():
            out = self(src)
        # get classes if classification
        if self.conf['nclasses'] > 0:
            out = self.one_hot_to_classes(out)
        # return predictions and ground-truths
        return out, tgt


    def test_epoch_end(self, outputs):
        # concatenate outputs
        out = torch.cat([cur_out[0] for cur_out in outputs],0)
        tgt = torch.cat([cur_out[1] for cur_out in outputs],0)
        # invert
        out = self.tgt_norm.invert(out)
        tgt = self.tgt_norm.invert(tgt)
        # compute results
        if self.conf['nclasses'] > 0:
            cur_val = test_batch_classification(out, tgt, self.tgt_min, self.tgt_max, self.tgt_avg, self.conf['tgt_vars'])
        else:
            cur_val = test_batch_regression(out, tgt, self.tgt_min, self.tgt_max, self.tgt_avg, self.conf['tgt_vars'])
        # log results
        self.log_dict(cur_val)
        # return them
        return cur_val


    def predict_step(self, batch, batch_idx):
        # split
        src, tgt, coords = batch
        # compute prediction
        with torch.no_grad(): out = model(src)
        # project to output space
        out = self.tgt_norm.invert(out)
        # concatenate coordinates
        return torch.cat((coords.cpu(), out.cpu()),1)


    def on_predict_epoch_end(self, outputs):
        # concatenate all outputs
        res = torch.cat(outputs[0], axis=0)
        # create dataframe
        df = pd.DataFrame(data=res.numpy(), columns=['lat','lon'] + self.conf['tgt_vars'])
        # save as attribute for later use
        self.result = df
        # return
        return df


    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(),
                            weight_decay = self.conf['weight_decay'],
                            lr = self.conf['learning_rate'])


    def train_dataloader(self):
        # create dataset
        ds = DatasetLucas(self.conf, 'train_csv', \
                            self.src_norm, self.tgt_norm, self.binner)
        # save min max and avg
        self.tgt_min, self.tgt_max, self.tgt_avg = ds.get_min_max_avg()
        # create dataloader
        return torch.utils.data.DataLoader(ds, batch_size=self.conf['batch_size'], drop_last=True, \
                                            shuffle=True, num_workers=self.conf['num_workers'])


    def val_dataloader(self):
        # create dataset
        ds = DatasetLucas(self.conf, 'val_csv', \
                            self.src_norm, self.tgt_norm, self.binner)
        # create dataloader
        return torch.utils.data.DataLoader(ds, batch_size=self.conf['batch_size'], drop_last=False, \
                                            shuffle=False, num_workers=self.conf['num_workers'])


    def test_dataset(self, return_coords=False):
        # create dataset
        return DatasetLucas(self.conf, 'test_csv', \
                            self.src_norm, self.tgt_norm, self.binner, return_coords=return_coords)

    def test_dataloader(self, return_coords=False):
        # create dataset
        ds = self.test_dataset(return_coords=False)
        # return dataloader
        return torch.utils.data.DataLoader(ds, batch_size=self.conf['batch_size'], drop_last=False, \
                                            shuffle=False, num_workers=self.conf['num_workers'])

    def predict_dataloader(self, return_coords=False):
        # create dataset
        ds = self.test_dataset(return_coords=True)
        # return dataloader
        return torch.utils.data.DataLoader(ds, batch_size=self.conf['batch_size'], drop_last=False, \
                                            shuffle=False, num_workers=self.conf['num_workers'])


    def get_loss(self, loss):
        if loss == 'l1': return F.l1_loss
        if loss == 'l2' or loss == 'mse': return F.mse_loss
        if loss == 'cross_entropy': return self.classification_loss # nn.CrossEntropyLoss()


    def classification_loss(self, inp, tgt):
        # define number of classes
        nclasses = self.conf['nclasses']
        # define number of target variables
        ntargets = len(self.conf['tgt_vars'])
        # init loss container
        loss = 0
        # for each target
        for i in range(ntargets):
            # get range
            rng_srt, rng_end = i*nclasses, (i+1)*nclasses
            # apply softmax
            inp[:,rng_srt:rng_end] = F.log_softmax(inp[:,rng_srt:rng_end].clone(), dim=-1)
            # add current loss
            loss += F.cross_entropy(inp[:,rng_srt:rng_end], tgt[:,i].long())
        # return loss
        return loss



def parse_unknown_args(unk):
    # convert unknown args in key-val
    new_keys = [cur_key.replace('-','') for cur_key in unk[::2]]
    unk = dict(zip(new_keys, unk[1::2]))
    # parse
    for cur_key in unk:
        try:
            unk[cur_key] = int(unk[cur_key])
        except:
            try:
                unk[cur_key] = float(unk[cur_key])
            except:
                continue
    return unk



if __name__ == '__main__':
    # parse arguments
    # get arguments
    parser = argparse.ArgumentParser()
    # checkpoint path
    parser.add_argument("-json", "--json", help="Json file (only for training).",
                        default='', type=str)
    parser.add_argument("-ckp", "--checkpoint", help="Checkpoint If training, resumes.",
                        default=None, type=str)
    parser.add_argument("-map", "--map", help="Map shapefile. If empty, no rendering.",
                        default='', type=str)
    parser.add_argument("-crs", "--crs", help="CRS of map and coords in csv.",
                        default=4326, type=int)
    parser.add_argument("-sr", "--spatial_resolution", help="Map shapefile. If empty, no rendering.",
                        default=(0.05, 0.05), type=float, nargs=2)
    parser.add_argument("-sep", "--sep", help="Col separator.",
                        default=';', type=str)
    parser.add_argument("-out", "--out", help="Output filename.",
                        default='', type=str)
    parser.add_argument("-train", "--train", help="If set, trains the system.",
                        action='store_true')
    parser.add_argument("-test", "--test", help="If set, computes metrics instead of regen.",
                        action='store_true')
    # parse args
    args, further_args = parser.parse_known_args()

    # try to parse arguments as floats
    further_args = parse_unknown_args(further_args)
    
    # check if training or test/regen
    if args.train:
        # define output directory
        if args.out == '':
            args.out = os.path.splitext(args.json)[0]
        # create it
        os.makedirs(args.out, exist_ok=True)
        print('Weights will be saved in {}'.format(args.out))

        # if checkpoint is specified
        if args.checkpoint is not None and os.path.isfile(args.checkpoint):
            # load from checkpoint if specified
            # conf = torch.load(args.checkpoint)['hyper_parameters']
            print('Calling model with these added arguments:')
            pprint(further_args)
            model = Experiment.load_from_checkpoint(args.checkpoint, **further_args)
            conf = model.conf
            model.src_norm.setup[0] = True
            model.tgt_norm.setup[0] = True
        else:
            # else read json file containing the configuration
            with open(args.json) as f:
                conf = json.load(f)
            # and load model
            model = Experiment(conf)


        # callbacks = [EarlyStopping(monitor='r2/global', mode='max', patience=50)]
        
        # define callbacks
        if not 'nclasses' in conf or conf['nclasses'] == 0:
            # for regression
            # cb_names = [ 'r2', 'mae', 'mse', 'rmse', 'pearson']
            # cb_modes = ['max', 'min', 'min',  'min',     'max']
            cb_names = [ 'r2', 'r2_lin', 'rmse_min_max', 'rmse_avg']
            cb_modes = ['max',    'max',          'min',      'min']
        else:
            # for classification
            cb_names = ['accuracy', 'precision', 'recall',  'f1']
            cb_modes = [     'max',       'max',    'max', 'max']
        # init callbacks
        callbacks = [
            ModelCheckpoint(monitor=f'{cur_name}/avg', mode=cur_mode, save_top_k=1, save_last=True, filename=cur_name)
            for cur_name, cur_mode in zip(cb_names, cb_modes)
        ]
        callbacks.append(RichProgressBar())
        
        # define logger
        tb_logger = pl_loggers.TensorBoardLogger(args.out, name=None)

        # define number of epochs
        nepochs = conf['nepochs'] if 'nepochs' in conf else 2000
        
        # define trainer
        trainer = pl.Trainer(gpus = 1, max_epochs = nepochs,
                            # weights_save_path = args.out,
                            # progress_bar_refresh_rate = 20,
                            # track_grad_norm = False,
                            # auto_lr_find = True,
                            log_every_n_steps = 1,
                            callbacks = callbacks,
                            logger = tb_logger,
                            resume_from_checkpoint = args.checkpoint)
        # train
        trainer.fit(model)
        # load best
        ckpt_path = trainer.checkpoint_callback.best_model_path
        print(ckpt_path)
        
    else:
        
        # at this point, either test or regenerate
        # load model
        model = Experiment.load_from_checkpoint(args.checkpoint)
        model.eval()
        # if testing is set
        if args.test:
            if args.out == '':
                args.out = os.path.join(os.path.dirname(args.checkpoint), 'results.csv')
            # test metrics
            trainer = pl.Trainer(gpus=1, deterministic=True)
            res = trainer.test(model=model)
            res = res[0]
            # save to file
            df = pd.DataFrame.from_dict(res.items())
            df.columns = ['var', 'val']
            df = df.set_index('var')
            df.to_csv(args.out, sep=args.sep, float_format='%.4f')
        else:
            # if out not specified, save in folder
            if args.out == '':
                args.out = os.path.join(os.path.dirname(args.checkpoint), 'regen.csv')
            # define trainer
            trainer = pl.Trainer(gpus=1, deterministic=True)
            # perform predictions
            trainer.predict(model=model)
            # get result
            df = model.result
            # save to csv
            df.to_csv(args.out, sep=args.sep, float_format='%.6f', index=False)
            # render
            if not args.map == '':
                from Renderer import Renderer
                renderer = Renderer(args.map, crs=args.crs)
                renderer.render(df, model.conf['tgt_vars'], out_dir=os.path.dirname(args.out), \
                    spatial_res=args.spatial_resolution, lon_key='lon', lat_key='lat')

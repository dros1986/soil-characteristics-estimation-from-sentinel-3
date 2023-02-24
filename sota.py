import pandas as pd
import numpy as np
import argparse
import os
import pickle
import time
# import torch
# import matplotlib.pyplot as plt

# Models
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

# Metrics
from sklearn.metrics import r2_score

# Miscellaneous
from sklearn import utils
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import PredefinedSplit
from sklearn.multioutput import MultiOutputRegressor
from tqdm import tqdm

# from testing import test_batch


def get_src(df, src_prefix, fmin=None, fmax=None):
    ''' gets infrared signals using cols starting with src_prefix '''
    # get x points
    src_cols = [col for col in df.columns if col.startswith(src_prefix)]
    x = np.array([float(col[len(src_prefix):]) for col in src_cols])
    # sort x points in increasing order
    pos = np.argsort(x)
    src_cols = [src_cols[cur_pos] for cur_pos in pos]
    # extract x and y values
    src_x = x[pos]
    src_y = df[src_cols].to_numpy()
    # keep only frequences in range
    cond = np.ones(src_x.shape).astype(bool)
    if fmin is not None:
        # import ipdb; ipdb.set_trace()
        cond *= src_x>=fmin
    if fmax is not None:
        cond *= src_x<=fmax
    src_y = src_y[:,cond]
    src_cols = [cur_col for (cur_col, cur_cond) in zip(src_cols, cond) if cur_cond]
    src_x = src_x[cond]
    # convert to tensor
    # src_x = torch.from_numpy(src_x).float()
    # src_y = torch.from_numpy(src_y).float()
    src_x = src_x.astype(np.float32)
    src_y = src_y.astype(np.float32)
    # return
    return src_cols, src_x, src_y


def measure(y_pred, y_test):
    mae = np.mean(abs(y_test - y_pred))
    mse = np.mean((y_test - y_pred)**2)
    rmse = np.sqrt(np.mean((y_test - y_pred)**2))
    r2 = r2_score(y_test, y_pred)
    return mae, mse, rmse, r2

def measure_all(y_pred, y_test, var_names, model_name, elapsed_time):
    # create output
    df = pd.DataFrame(columns=['model', 'var', 'mae', 'mse', 'rmse', 'r2', 'time'])
    # for each variable
    for i,cur_var in enumerate(var_names):
        # compute measures
        mae, mse, rmse, r2 = measure(y_pred[:,i], y_test[:,i])
        # append result
        # df = df.append({'var':cur_var, 'mae':mae, 'mse':mse, 'rmse':rmse, 'r2':r2}, ignore_index=True)
        df = pd.concat([df, pd.DataFrame.from_records([{'model': model_name, 'var':cur_var, 'mae':mae, 'mse':mse, 'rmse':rmse, 'r2':r2, 'time':elapsed_time}])])
    # return
    return df


################################################################################

if __name__ == '__main__':
    # parse arguments
    # get arguments
    parser = argparse.ArgumentParser()
    # checkpoint path
    parser.add_argument("-indir", "--indir", help="Input dir.",
    					default='/home/flavio/datasets/LucasLibrary/shared/', type=str)
    parser.add_argument("-outdir", "--outdir", help="Output dir.",
    					default='./sota_multioutput', type=str)
    # parser.add_argument("-train", "--train", help="If set, trains models.",
    # 					action='store_true')
    parser.add_argument("-render", "--render", help="If set, renders points in map.",
    					action='store_true')
    parser.add_argument("-map", "--map", help="Map.",
    					default='/home/flavio/Scaricati/ref-nuts-2021-01m.shp/NUTS_RG_01M_2021_4326.shp/NUTS_RG_01M_2021_4326.shp',
                        type=str)
    parser.add_argument("-tgt_vars", "--tgt_vars", help="Variables.",
    					default = ['coarse','clay','silt','sand','pH.in.CaCl2','pH.in.H2O','OC','CaCO3','N','P','K','CEC'],
                        nargs='+')
    parser.add_argument("-models", "--models", help="Models.",
    					default = ['rf','svr','brt'],
                        nargs='+')
    parser.add_argument("-fmin", "--fmin", help="Lowest band.",
                        default=None, type=float)
    parser.add_argument("-fmax", "--fmax", help="Highest band.",
                        default=None, type=float)
    # parse args
    args = parser.parse_args()

    # create output dir
    os.makedirs(args.outdir, exist_ok=True)

    # create output
    res = None
    # if args.train:
    # load training data
    # print("Reading train data...")
    lucas_train = pd.read_csv(os.path.join(args.indir, "lucas_dataset_train.csv"), sep=',')
    lucas_test = pd.read_csv(os.path.join(args.indir, "lucas_dataset_test.csv"))
    # lucas_train = pd.read_csv(os.path.join(args.indir, "train_real.csv"), sep=',')
    # lucas_test = pd.read_csv(os.path.join(args.indir, "test.csv"))
    # for each model
    for model_name in tqdm(args.models):
        # get training data
        cols_train, bands_train, x_train = get_src(lucas_train, 'spc.', fmin=args.fmin, fmax=args.fmax)
        cols_test, bands_test, x_test = get_src(lucas_test, 'spc.', fmin=args.fmin, fmax=args.fmax)
        y_train = np.array(lucas_train[args.tgt_vars])
        y_test = np.array(lucas_test[args.tgt_vars])
        # define model
        if model_name == 'rf':
            model = RandomForestRegressor(max_features = y_train.shape[1], n_estimators = 200, random_state=123, verbose=2)
            # model = RandomForestRegressor(max_features = y_train.shape[1], n_estimators = 2, random_state=123, verbose=2)
        elif model_name == 'svr':
            model = svm.SVR(kernel = "rbf", C = 1000, gamma = 0.01, verbose=2)
            # model = RandomForestRegressor(max_features = y_train.shape[1], n_estimators = 2, random_state=123, verbose=2)
        else:
            model = GradientBoostingRegressor(learning_rate = 0.1, min_samples_split = 6, n_estimators = 200, random_state=123, verbose=2)
            # model = GradientBoostingRegressor(learning_rate = 0.1, min_samples_split = 6, n_estimators = 2, random_state=123, verbose=2)
        # create multioutput model
        predictor = MultiOutputRegressor(model, n_jobs=8).fit(x_train, y_train)
        # predict
        start = time.time()
        y_pred = predictor.predict(x_test)
        elapsed_time = time.time() - start
        # measure
        cur_res = measure_all(y_pred, y_test, args.tgt_vars, model_name, elapsed_time)
        # append
        if res is None:
            res = cur_res
        else:
            res = pd.concat([res, cur_res], axis=0)
        # save predictor
        cur_fn = os.path.join(args.outdir, f'{model_name}.pkl')
        pickle.dump(predictor, open(cur_fn, 'wb'))
        # # create latex of metric R2
        # r2_latex = cur_res[['var', 'r2']]
        # r2_latex = r2_latex.set_index('var')
        # r2_latex.loc['avg'] = r2_latex['r2'].mean()
        # # save it as latex
        # r2_latex.T.to_latex(os.path.join(args.outdir,"{}.tex".format(model_name)), float_format='%.4f', decimal='.', index=True)
    # save global result
    res.to_csv(os.path.join(args.outdir,"all.csv"), index=False)
    res.to_excel(os.path.join(args.outdir,"all.xlsx"), index=False)
    # save r2 as latex
    r2_latex = res[['model', 'var', 'r2']]
    r2_latex = pd.concat([r2_latex[r2_latex['model']==cur_model][['var', 'r2']].set_index('var').T for cur_model in args.models])
    r2_latex['avg'] = r2_latex.mean(1)
    r2_latex.insert(0, 'model', args.models)
    r2_latex.to_latex(os.path.join(args.outdir,"r2.tex"), float_format='%.4f', decimal='.', index=False)
    print('hey')
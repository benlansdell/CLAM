from __future__ import print_function

import argparse
import pdb
import os
import math

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils import train
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np

import ray
from ray import tune
from datetime import datetime

from cnvrgcallback import CNVRGCallback
import cnvrg

from topk import SmoothTop1SVM

try:
    #ray.init(address='auto')
    ray.init(address="localhost:6379",ignore_reinit_error=True,log_to_driver=False)
except ConnectionError:
    print("Couldn't find a ray head node to connect to. Starting a local node")
    ray.init()

#CUDA_VISIBLE_DEVICES=NNN /home/abbas/anaconda3/envs/clam/bin/python /home/abbas/CLAM/main.py
#  --drop_out --lr 1e-4 --reg 1e-4 --k 10 --max_epochs 200 --label_frac 1 --k_start SSS --k_end EEE 
#  --exp_code task_2_tumor_subtyping_1024_lr1e-4_reg1e-4_adamw_CLAM_100 --weighted_sample --bag_loss ce 
#  --inst_loss svm --task task_2_tumor_subtyping --split_dir /home/abbas/CLAM/splits/task_2_tumor_subtyping_100/ 
#  --model_type clam_sb --log_data --subtyping --data_root_dir /mnt/storage/COMET/preprocessed_test1024_fp/features/ &'

#Update main.py and main_ray_tune.py with path for features...

"""
pip3 install torch==1.9.1+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html 
pip install ray[default]
pip install ray[tune]

git clone https://github.com/oval-group/smooth-topk.git
cd smooth-topk
pip install -e .
cd ..
"""

#CNVRG test run
"""
python main.py \
 --drop_out --lr 1e-4 --reg 1e-4 --k 10 --max_epochs 10 --label_frac 1 --k_start 0 --k_end 1 --early_stopping \
 --exp_code task_2_tumor_subtyping_1024_lr1e-4_reg1e-4_adamw_CLAM_100 --weighted_sample --bag_loss ce \
 --inst_loss svm --task task_2_tumor_subtyping \
 --model_type clam_sb --log_data --subtyping \
 --save_activations \
 --split_dir /cnvrg/splits/task_2_tumor_subtyping_100/ \
 --data_root_dir /data/comet_rms/preprocessed_test1024_fp/features/
"""
# --data_root_dir /mnt/storage/COMET/preprocessed_test1024_fp/features/

#cnvrg main run
"""
python main_ray_tune_cnvrg.py \
 --drop_out --lr 1e-4 --reg 1e-4 --k 10 --max_epochs 100 --label_frac 1 --k_start 0 --k_end 10 --early_stopping \
 --exp_code task_2_tumor_subtyping_1024_lr1e-4_reg1e-4_adamw_CLAM_100 --weighted_sample --bag_loss ce \
 --inst_loss svm --task task_2_tumor_subtyping --split_dir /cnvrg/splits/task_2_tumor_subtyping_100 \
 --model_type clam_sb --log_data --subtyping --data_root_dir /data/comet_rms/preprocessed_test1024_fp/features/
 """

#Short test run
"""
CUDA_VISIBLE_DEVICES=0 /home/blansdell/anaconda3/envs/clam/bin/python /home/blansdell/projects/comet/CLAM/main.py \
 --drop_out --lr 1e-4 --reg 1e-4 --k 10 --max_epochs 10 --label_frac 1 --k_start 0 --k_end 1 --early_stopping \
 --exp_code task_2_tumor_subtyping_1024_lr1e-4_reg1e-4_adamw_CLAM_100 --weighted_sample --bag_loss ce \
 --inst_loss svm --task task_2_tumor_subtyping --split_dir /home/abbas/CLAM/splits/task_2_tumor_subtyping_100/ \
 --model_type clam_sb --log_data --subtyping --data_root_dir /mnt/storage/COMET/preprocessed_test1024_fp/features/ \
 --save_activations
"""

#Base run
"""
CUDA_VISIBLE_DEVICES=0 /home/blansdell/anaconda3/envs/clam/bin/python /home/blansdell/projects/comet/CLAM/main.py \
 --drop_out --lr 1e-4 --reg 1e-4 --k 10 --max_epochs 100 --label_frac 1 --k_start 0 --k_end 10 --early_stopping \
 --exp_code task_2_tumor_subtyping_1024_lr1e-4_reg1e-4_adamw_CLAM_100 --weighted_sample --bag_loss ce \
 --inst_loss svm --task task_2_tumor_subtyping --split_dir /home/abbas/CLAM/splits/task_2_tumor_subtyping_100/ \
 --model_type clam_sb --log_data --subtyping --data_root_dir /mnt/storage/COMET/preprocessed_test1024_fp/features/ \
 --save_activations
"""

#Ray tune run
"""
CUDA_VISIBLE_DEVICES=0 /home/blansdell/anaconda3/envs/clam/bin/python /home/blansdell/projects/comet/CLAM/main_ray_tune.py \
 --drop_out --lr 1e-4 --reg 1e-4 --k 10 --max_epochs 100 --label_frac 1 --k_start 0 --k_end 10 --early_stopping \
 --exp_code task_2_tumor_subtyping_1024_lr1e-4_reg1e-4_adamw_CLAM_100 --weighted_sample --bag_loss ce \
 --inst_loss svm --task task_2_tumor_subtyping --split_dir /home/abbas/CLAM/splits/task_2_tumor_subtyping_100/ \
 --model_type clam_sb --log_data --subtyping --data_root_dir /mnt/storage/COMET/preprocessed_test1024_fp/features/
"""

#Evaluation code
"""
CUDA_VISIBLE_DEVICES=0 python eval.py --drop_out --k 10 --models_exp_code task_2_tumor_subtyping_CLAM_50_s1 \
                                      --save_exp_code task_2_tumor_subtyping_1024_lr1e-4_reg1e-4_adamw_CLAM_100 --task task_2_tumor_subtyping \
                                      --models_exp_code task_2_tumor_subtyping_1024_lr1e-4_reg1e-4_adamw_CLAM_100_s1 --model_type clam_sb \
                                      --results_dir ~/projects/comet/CLAM/results \
                                      --data_root_dir /mnt/storage/COMET/preprocessed_test1024_fp/features/ \
                                      --splits_dir /home/abbas/CLAM/splits/task_2_tumor_subtyping_100/
"""

#Heatmap code
"""
CUDA_VISIBLE_DEVICES=0,1 python create_heatmaps.py --config config_comet_1024_CLAM_100_s1.yaml
"""

#Questions:
# What do attention maps look like?
# What do mistakes look like? Plot them
# Where are the labels?
# What other parameters could be played with?
# How many patches are here, in total, compared to what is in their paper?


def main(args):

    ############################################################################

    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.split_dir is None:
        args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))
    #else:
    #    args.split_dir = os.path.join('splits', args.split_dir)

    print('split_dir: ', args.split_dir)
    assert os.path.isdir(args.split_dir)

    settings.update({'split_dir': args.split_dir})

    with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
        print(settings, file=f)
    f.close()

    print("################# Settings ###################")
    for key, val in settings.items():
        print("{}:  {}".format(key, val))

    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    folds = np.arange(start, end)
    for i in folds:
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, 
                csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        
        datasets = (train_dataset, val_dataset, test_dataset)
        results, test_auc, val_auc, test_acc, val_acc  = train(datasets, i, args)
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        #tune.report(train_auc = train_auc)
        #tune.report(train_acc = train_acc)
        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    tune.report(val_auc = np.mean(val_auc))
    tune.report(val_acc = np.mean(val_acc))
    tune.report(test_auc = np.mean(test_auc))
    tune.report(test_acc = np.mean(test_acc))

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 
        'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc})

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None, 
                    help='data directory')
parser.add_argument('--max_epochs', type=int, default=200,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--save_activations', default=False,
                    action = 'store_true', help = 'Whether to save activations at end of training (default: False)')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None, 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', action='store_true', default=False, help='enabel dropout (p=0.25)')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce',
                     help='slide-level classification loss function (default: ce)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb', 
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='size of model, does not affect mil')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping'])
### CLAM specific options
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                     help='disable instance-level clustering')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
                     help='instance-level clustering loss function (default: None)')
parser.add_argument('--subtyping', action='store_true', default=False, 
                     help='subtyping problem')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')
args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

encoding_size = 1024
settings = {'num_splits': args.k, 
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs, 
            'results_dir': args.results_dir, 
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size': args.model_size,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt}

if args.model_type in ['clam_sb', 'clam_mb']:
   settings.update({'bag_weight': args.bag_weight,
                    'inst_loss': args.inst_loss,
                    'B': args.B})

print('\nLoad Dataset')

if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_vs_normal_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_vs_normal_resnet_features'),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=3
    dataset = Generic_MIL_Dataset(csv_path = 'cometFiltered.csv',
                            data_dir= '/data/comet_rms/preprocessed_test1024_fp/features/',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'A_':0, 'E_':1, 'S_':2},
                            patient_strat= False,
                            ignore=[])

    if args.model_type in ['clam_sb', 'clam_mb']:
        assert args.subtyping 
        
else:
    raise NotImplementedError
    

##Setup the ray tune run:
n_samples = 200
sweep = {}
sweep['lr'] = tune.loguniform(1e-5, 1e-2)
sweep['reg'] = tune.loguniform(1e-6, 1e-3)
sweep['bag_loss'] = tune.choice(['ce', 'svm'])
sweep['model_type'] = tune.choice(['clam_sb', 'clam_mb'])
sweep['model_size'] = tune.choice(['small', 'big'])
sweep['drop_out'] = tune.choice([True, False])

def train_func(args, cfg):
    args.lr = cfg['lr']
    args.reg = cfg['reg']
    args.bag_loss = cfg['bag_loss']
    args.model_type = cfg['model_type']
    args.model_size = cfg['model_size']
    args.drop_out = cfg['drop_out']
    results = main(args)

tracked_metrics = ['val_auc', 'val_err']

if __name__ == "__main__":

    sweep_name = 'baseline_sweep'
    name = sweep_name + '_' + datetime.now().strftime("%d-%m-%y_%H-%M-%S")
    fn_results = name + '_tune_results.csv'

    analysis = tune.run(
        lambda x: train_func(args, x),
        name = name,
        config = sweep,
        num_samples = n_samples,
        resources_per_trial={"cpu": 1, "gpu": 1},
        max_failures = 2,
        callbacks=[CNVRGCallback(tracked_metrics)])

    print("Best config: ", analysis.get_best_config(metric="val_auc", mode="max"))

    # Get a dataframe for analyzing trial results.
    analysis.results_df.to_csv(fn_results)
        
    print("finished!")
    print("end script")
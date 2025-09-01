import warnings
warnings.filterwarnings('ignore')
from rdkit import RDLogger  
RDLogger.DisableLog('rdApp.*')  
from argparse import Namespace
from logging import Logger
import os
from typing import Tuple

import numpy as np

from chemprop.train.run_training import run_training
from chemprop.data.utils import get_task_names
from chemprop.utils import makedirs
from chemprop.parsing import parse_train_args, modify_train_args
from chemprop.utils import create_logger
from chemprop.parsing import parse_predict_args
from chemprop.train import make_predictions
import torch

def cross_validate(args: Namespace, logger: Logger = None) -> Tuple[float, float]:
    """k-fold cross validation"""
    info = logger.info if logger is not None else print

    # Initialize relevant variables
    init_seed = args.seed
    save_dir = args.save_dir
    task_names = get_task_names(args.data_path)

    # Decide which label to use in logs (no test -> 'val', otherwise 'test')
    sep_test = getattr(args, 'separate_test_path', None)
    split_sizes = getattr(args, 'split_sizes', None)
    try:
        has_test = (sep_test is not None and str(sep_test).strip() != '') or \
                   (split_sizes is not None and len(split_sizes) >= 3 and float(split_sizes[2]) > 0.0)
    except Exception:
        has_test = False
    eval_label = 'test' if has_test else 'val'

    # Run training on different random seeds for each fold
    all_scores = []
    for fold_num in range(args.num_folds):
        info(f'Fold {fold_num}')
        args.seed = init_seed + fold_num
        args.save_dir = os.path.join(save_dir, f'fold_{fold_num}')
        makedirs(args.save_dir)
        model_scores = run_training(args, logger)
        all_scores.append(model_scores)
    all_scores = np.array(all_scores)

    # Report results
    info(f'{args.num_folds}-fold cross validation')

    # Report scores for each fold
    for fold_num, scores in enumerate(all_scores):
        info(f'Seed {init_seed + fold_num} ==> {eval_label} {args.metric} = {np.nanmean(scores):.6f}')

        if args.show_individual_scores:
            for task_name, score in zip(task_names, scores):
                info(f'Seed {init_seed + fold_num} ==> {eval_label} {task_name} {args.metric} = {score:.6f}')

    # Report scores across models
    avg_scores = np.nanmean(all_scores, axis=1)  # average score for each model across tasks
    mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)
    info(f'Overall {eval_label} {args.metric} = {mean_score:.6f} +/- {std_score:.6f}')

    if args.show_individual_scores:
        for task_num, task_name in enumerate(task_names):
            info(f'Overall {eval_label} {task_name} {args.metric} = '
                 f'{np.nanmean(all_scores[:, task_num]):.6f} +/- {np.nanstd(all_scores[:, task_num]):.6f}')

    return mean_score, std_score


if __name__ == '__main__':
    n_repeat = 1
    score_lst = []
    
    for i_repeat in range(n_repeat):
        args = parse_train_args()
        use_gpu = torch.cuda.is_available()
        args.no_cuda = not use_gpu
        args.gpu = 0 if use_gpu else None

        # DATA
        args.data_path = './data/CMPNN3_filtered.csv' # file để train
        args.dataset_type = 'classification'
        args.split_type = 'random'
        args.split_sizes = [0.7, 0.1, 0.2] # split train/val/test

        # TRAINING / EVALUATION
        args.num_folds = 5
        args.epochs = 30
        args.ensemble_size = 1
        args.batch_size = 64
        args.metric = 'auc'
        args.extra_metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        # SEED & SAVE
        # args.seed += i_repeat * 100
        args.save_dir = './ckpt/cv5'
        
        modify_train_args(args)
        print('Using device:', 'cuda' if args.cuda else 'cpu')
        logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)

        mean_auc_score, std_auc_score = cross_validate(args, logger)
        score_lst.append(mean_auc_score)
    
    print(score_lst)
    print(n_repeat, f'repeats score: {np.nanmean(score_lst):.5f} +/- {np.nanstd(score_lst):.5f}')
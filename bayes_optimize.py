import csv
import os
import pickle
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from chemprop.parsing import parse_train_args, modify_train_args
from train import cross_validate
import torch


CKPT_DIR = 'ckpt'
HYPEROPT_DIR = os.path.join(CKPT_DIR, 'hyperopt')
TRIALS_PKL = os.path.join(CKPT_DIR, 'trials.pkl')
RESULTS_DIR = 'results'
BAYES_RESULTS_CSV = os.path.join(RESULTS_DIR, 'bayes_optimize_results.csv')

os.makedirs(HYPEROPT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


search_space = {
    'dropout': hp.uniform('dropout', 0.0, 0.5), # tỉ lệ bỏ học
    'depth': hp.choice('depth', [3, 4, 5]), # số bước truyền tin
    'hidden_size': hp.choice('hidden_size', [300, 350, 400, 450, 500, 550, 600]), # số lượng nơ-ron trong lớp ẩn
    'ffn_num_layers': hp.choice('ffn_num_layers', [2, 3, 4, 5]) # số lớp ẩn của FFN
}


MAX_EVALS = 120   # khảo sát 120 bộ tham số (120 model)
TUNING_SEED = 42  # cố định split để các trial công bằng


def objective(params):
    args = parse_train_args()
    use_gpu = torch.cuda.is_available()
    args.no_cuda = not use_gpu
    args.gpu = 0 if use_gpu else None

    # DATA & SPLIT FOR TUNING: train/val = 90/10, KHÔNG test
    args.data_path = './data/CMPNN3_filtered.csv'
    args.dataset_type = 'classification'
    args.split_type = 'random'
    args.split_sizes = [0.9, 0.1, 0.0]
    args.num_folds = 1
    args.seed = TUNING_SEED

    # METRICS
    args.metric = 'auc'
    args.extra_metrics = ['accuracy', 'precision', 'recall', 'f1']

    # HYPERPARAMETERS
    args.dropout = params['dropout']
    args.depth = params['depth']
    args.hidden_size = params['hidden_size']
    args.ffn_num_layers = params['ffn_num_layers']

    # SAVE DIR RIÊNG CHO MỖI TRIAL (tránh ghi đè)
    trial_id = len(trials.trials)
    args.save_dir = os.path.join(HYPEROPT_DIR, f'trial_{trial_id}')

    modify_train_args(args)
    print('Using device:', 'cuda' if args.cuda else 'cpu')

    auc, _ = cross_validate(args)
    return {'loss': -auc, 'status': STATUS_OK, 'params': params, 'auc': auc}


def save_progress(trials: Trials) -> None:
    """Persist trials and write results to CSV."""

    with open(TRIALS_PKL, 'wb') as fp:
        pickle.dump(trials, fp)

    with open(BAYES_RESULTS_CSV, 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(
            ['dropout', 'depth', 'hidden_size', 'ffn_num_layers', 'val_AUC']
        )
        for trial in trials.trials:
            params = trial['result']['params']
            writer.writerow(
                [
                    params['dropout'],
                    params['depth'],
                    params['hidden_size'],
                    params['ffn_num_layers'],
                    trial['result']['auc'],
                ]
            )


# Resume trials (nếu có) 
if os.path.exists(TRIALS_PKL):
    with open(TRIALS_PKL, 'rb') as fp:
        trials: Trials = pickle.load(fp)
    print(f"Loaded {len(trials.trials)} previous trials")
else:
    trials = Trials()


start_eval = len(trials.trials)
for i in range(start_eval, MAX_EVALS):
    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=i + 1,
        trials=trials,
        show_progressbar=True,
    )
    save_progress(trials)
    best_val_auc = max(t['result']['auc'] for t in trials.trials)
    print(f"Trial {i + 1}/{MAX_EVALS} complete - Best AUC: {best_val_auc:.4f}")


best_trial = max(trials.trials, key=lambda t: t['result']['auc'])
print('Best hyperparameters:', best_trial['result']['params'])
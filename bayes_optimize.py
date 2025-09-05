import csv
import os
import pickle
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from chemprop.parsing import parse_train_args, modify_train_args
from train import cross_validate
import torch


search_space = {
    'dropout': hp.uniform('dropout', 0.0, 0.5), # tỉ lệ bỏ học
    'depth': hp.choice('depth', [3, 4, 5]), # số bước truyền tin
    'hidden_size': hp.choice('hidden_size', [300, 350, 400, 450, 500, 550, 600]), # số lượng nơ-ron trong lớp ẩn
    'ffn_num_layers': hp.choice('ffn_num_layers', [2, 3, 4, 5]) # số lớp ẩn của FFN
}


MAX_EVALS = 120  # khảo sát 120 bộ tham số (120 model)


def objective(params):
    args = parse_train_args()
    use_gpu = torch.cuda.is_available()
    args.no_cuda = not use_gpu
    args.gpu = 0 if use_gpu else None
    args.dataset_type = 'classification'
    args.metric = 'auc'
    args.data_path = './data/CMPNN3_filtered.csv'
    args.split_sizes = [0.9, 0.1, 0.0] # dùng validation AUC -> bỏ test set
    args.dropout = params['dropout']
    args.depth = params['depth']
    args.hidden_size = params['hidden_size']
    args.ffn_num_layers = params['ffn_num_layers']
    modify_train_args(args)
    print('Using device:', 'cuda' if args.cuda else 'cpu')
    val_auc, _ = cross_validate(args)
    return {'loss': -val_auc, 'status': STATUS_OK, 'params': params, 'auc': val_auc}


def save_progress(trials: Trials) -> None:
    """Persist trials and write results to CSV."""

    with open('bayesian_trials.pkl', 'wb') as fp:
        pickle.dump(trials, fp)

    with open('bayesian_optimization_results.csv', 'w', newline='') as fp:
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


if os.path.exists('bayesian_trials.pkl'):
    with open('bayesian_trials.pkl', 'rb') as fp:
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
    best_auc = max(t['result']['auc'] for t in trials.trials)
    print(f"Trial {i + 1}/{MAX_EVALS} complete - Best AUC: {best_auc:.4f}")


best_trial = max(trials.trials, key=lambda t: t['result']['auc'])
print('Best hyperparameters:', best_trial['result']['params'])
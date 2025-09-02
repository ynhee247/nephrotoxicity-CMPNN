import torch
import csv
import os
import pickle
import pandas as pd
import numpy as np
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from chemprop.parsing import parse_train_args, modify_train_args, parse_predict_args
from chemprop.train.run_training import run_training
from chemprop.train import make_predictions
from chemprop.data.utils import get_task_names
from sklearn.metrics import roc_auc_score


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

    # SAVE SPLITS
    os.makedirs(args.save_dir, exist_ok=True)
    args.save_smiles_splits = True

    args.epochs = 30
    args.batch_size = 64
    args.ensemble_size = 1
    modify_train_args(args)
    print('Using device:', 'cuda' if args.cuda else 'cpu')

    # COMPUTE val_AUC (manually)
    # 1) Train (bỏ qua giá trị trả về vì đó là test score)
    _ = run_training(args, logger=None)   # train; test score (if any) is ignored
    
    # 2) Get validation SMILES from splits.csv
    val_list_path = os.path.join(args.save_dir, 'val_smiles.csv')
    if not os.path.exists(val_list_path):
        raise FileNotFoundError(f"Missing {val_list_path}. Did save_smiles_splits fail?")

    val_df = pd.read_csv(val_list_path)
    if val_df.shape[1] == 1:
        val_df.columns = ['smiles']
    if 'smiles' not in val_df.columns:
        val_df.columns = ['smiles']

    val_smiles = set(val_df['smiles'].astype(str))

    # 3) Build val.csv from original data
    df_all = pd.read_csv(args.data_path)
    df_val = df_all[df_all['smiles'].isin(val_smiles)].copy()
    assert len(df_val) > 0, "Validation set is empty. Check val_smiles.csv and 'smiles' column."
    val_csv = os.path.join(args.save_dir, 'val.csv')
    df_val.to_csv(val_csv, index=False)

    # 4) Predict on validation using best checkpoint in args.save_dir
    # Find the exact checkpoint (.pt)
    ckpt_dir = args.save_dir
    if os.path.isdir(os.path.join(args.save_dir, 'model_0')):
        ckpt_dir = os.path.join(args.save_dir, 'model_0')

    p_args = parse_predict_args()
    p_args.test_path = val_csv
    p_args.checkpoint_dir = ckpt_dir
    p_args.preds_path = None

    # Set device/batch_size
    use_gpu = torch.cuda.is_available()
    if hasattr(p_args, 'no_cuda'):
        p_args.no_cuda = not use_gpu
    if hasattr(p_args, 'gpu') and use_gpu:
        p_args.gpu = 0
    if hasattr(p_args, 'batch_size'):
        p_args.batch_size = getattr(args, 'batch_size', 64)
    if hasattr(p_args, 'smiles_column'):
        p_args.smiles_column = 'smiles'
    if hasattr(p_args, 'dataset_type'):
        p_args.dataset_type = 'classification'
    
    make_predictions(p_args)
    preds = np.asarray(preds)
    y_pred = preds[:, 0] if preds.ndim == 2 else preds.reshape(-1)

    # 5) Compute val_AUC
    task = get_task_names(args.data_path)[0]
    y_true = df_val[task].values
    assert len(y_true) == len(y_pred), f"Length mismatch: y_true={len(y_true)} vs y_pred={len(y_pred)}"

    val_auc = float(roc_auc_score(y_true, y_pred))

    return {'loss': -val_auc, 'status': STATUS_OK, 'params': params, 'auc': val_auc}


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
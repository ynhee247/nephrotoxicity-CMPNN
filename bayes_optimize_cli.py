import os, csv, pickle, subprocess, sys, json, shutil
import numpy as np
import pandas as pd
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.metrics import roc_auc_score


# Paths
DATA_CSV = './data/CMPNN3_filtered.csv'
CKPT_DIR = 'ckpt'
HOPT_DIR = os.path.join(CKPT_DIR, 'hyperopt_cli')
RESULTS_DIR = 'results'
TRIALS_PKL = os.path.join(CKPT_DIR, 'trials.pkl')
RESULTS_CSV = os.path.join(RESULTS_DIR, 'bayes_optimize_results.csv')

os.makedirs(HOPT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# Search space
search_space = {
    'dropout': hp.uniform('dropout', 0.0, 0.5), # tỉ lệ bỏ học
    'depth': hp.choice('depth', [3, 4, 5]), # số bước truyền tin
    'hidden_size': hp.choice('hidden_size', [300, 350, 400, 450, 500, 550, 600]), # số lượng nơ-ron trong lớp ẩn
    'ffn_num_layers': hp.choice('ffn_num_layers', [2, 3, 4, 5]) # số lớp ẩn của FFN
}

MAX_EVALS = 120   # khảo sát 120 bộ tham số (120 model)
TUNING_SEED = 42  # cố định split để các trial công bằng


def run(cmd, cwd=None):
    """Run a shell command; raise with readable message if fails."""
    print(">>", " ".join(cmd))
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(p.stdout)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}")
    return p.stdout


def chemprop_train_cli(trial_dir, params):
    """Train 1 trial via CLI (90/10/0, no test); write splits + checkpoint."""
    os.makedirs(trial_dir, exist_ok=True)
    cmd = [
        'chemprop_train',
        '--data_path', DATA_CSV,
        '--dataset_type', 'classification',
        '--split_type', 'random',
        '--split_sizes', '0.9', '0.1', '0.0', # train/val/test
        '--num_folds', '1',
        '--seed', str(TUNING_SEED),
        '--save_dir', trial_dir,
        '--save_smiles_splits',
        '--epochs', '30',
        '--batch_size', '64',
        '--ensemble_size', '1',
        '--metric', 'auc',
        '--dropout', str(params['dropout']),
        '--depth', str(params['depth']),
        '--hidden_size', str(params['hidden_size']),
        '--ffn_num_layers', str(params['ffn_num_layers']),
        '--quiet'
    ]
    # Fallback nếu entrypoint không tồn tại
    try:
        run(cmd)
    except Exception:
        cmd[0:1] = [sys.executable, '-m', 'chemprop.train']
        run(cmd)


def chemprop_predict_cli(checkpoint_dir, test_csv, preds_csv):
    """Predict via CLI; write preds_csv."""
    cmd = [
        'chemprop_predict',
        '--test_path', test_csv,
        '--checkpoint_dir', checkpoint_dir,
        '--preds_path', preds_csv,
        '--batch_size', '64'
    ]
    try:
        run(cmd)
    except Exception:
        cmd[0:1] = [sys.executable, '-m', 'chemprop.predict']
        run(cmd)
    if not os.path.exists(preds_csv):
        raise FileNotFoundError(f"Pred file not created: {preds_csv}")


def find_ckpt_dir(trial_dir):
    """Return dir that actually contains model .pt files."""
    m0 = os.path.join(trial_dir, 'model_0')
    return m0 if os.path.isdir(m0) else trial_dir


def objective(params):
    trial_id = len(trials.trials)
    trial_dir = os.path.join(HOPT_DIR, f'trial_{trial_id}')
    # 1) Train
    chemprop_train_cli(trial_dir, params)

    # 2) Build val.csv from val_smiles.csv
    val_smiles_path = os.path.join(trial_dir, 'val_smiles.csv')
    if not os.path.exists(val_smiles_path):
        raise FileNotFoundError(f'Missing {val_smiles_path}')
    val_smiles_df = pd.read_csv(val_smiles_path, header=None)
    if val_smiles_df.shape[1] == 1:
        val_smiles_df.columns = ['smiles']
    if 'smiles' not in val_smiles_df.columns:
        val_smiles_df.columns = ['smiles']
    val_smiles = set(val_smiles_df['smiles'].astype(str))

    df_all = pd.read_csv(DATA_CSV)
    if 'smiles' not in df_all.columns and 'SMILES' in df_all.columns:
        df_all = df_all.rename(columns={'SMILES': 'smiles'})
    df_val = df_all[df_all['smiles'].astype(str).isin(val_smiles)].copy()
    assert len(df_val) > 0, "Validation set is empty."
    val_csv = os.path.join(trial_dir, 'val.csv')
    df_val.to_csv(val_csv, index=False)

    # 3) Predict on validation
    preds_csv = os.path.join(trial_dir, 'val_preds.csv')
    ckpt_dir = find_ckpt_dir(trial_dir)
    chemprop_predict_cli(ckpt_dir, val_csv, preds_csv)

    # 4) Compute val AUC
    task_col = [c for c in df_all.columns if c != 'smiles'][0]   # 1-task binary
    preds_df = pd.read_csv(preds_csv)
    y_true = df_val[task_col].values
    y_pred = preds_df[task_col].values if task_col in preds_df.columns else preds_df.iloc[:, -1].values
    val_auc = float(roc_auc_score(y_true, y_pred))

    return {'loss': -val_auc, 'status': STATUS_OK, 'params': params, 'auc': val_auc, 'trial_dir': trial_dir}


def save_progress(trials: Trials):
    with open(TRIALS_PKL, 'wb') as f:
        pickle.dump(trials, f)
    with open(RESULTS_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['trial_dir','dropout','depth','hidden_size','ffn_num_layers','val_AUC'])
        for t in trials.trials:
            p = t['result']['params']
            w.writerow([t['result'].get('trial_dir',''),
                        p['dropout'], p['depth'], p['hidden_size'], p['ffn_num_layers'],
                        t['result']['auc']])


# Resume
if os.path.exists(TRIALS_PKL):
    with open(TRIALS_PKL, 'rb') as f:
        trials = pickle.load(f)
    print(f"Loaded {len(trials.trials)} previous trials")
else:
    trials = Trials()


# Main loop
start = len(trials.trials)
for i in range(start, MAX_EVALS):
    fmin(fn=objective, space=search_space, algo=tpe.suggest,
         max_evals=i+1, trials=trials, show_progressbar=True)
    save_progress(trials)
    best = max(t['result']['auc'] for t in trials.trials)
    print(f"Trial {i+1}/{MAX_EVALS} complete - Best val_AUC: {best:.4f}")


best_trial = max(trials.trials, key=lambda t: t['result']['auc'])
print('Best hyperparameters:', best_trial['result']['params'])
print('Best trial dir:', best_trial['result']['trial_dir'])
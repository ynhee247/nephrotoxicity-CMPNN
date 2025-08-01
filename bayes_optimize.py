import csv
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

trials = Trials()


def objective(params):
    args = parse_train_args()
    use_gpu = torch.cuda.is_available()
    args.no_cuda = not use_gpu
    args.gpu = 0 if use_gpu else None
    args.dataset_type = 'classification'
    args.metric = 'auc'
    args.data_path = './data/CMPNN3_filtered.csv'
    args.dropout = params['dropout']
    args.depth = params['depth']
    args.hidden_size = params['hidden_size']
    args.ffn_num_layers = params['ffn_num_layers']
    modify_train_args(args)
    print('Using device:', 'cuda' if args.cuda else 'cpu')
    auc, _ = cross_validate(args)
    return {'loss': -auc, 'status': STATUS_OK, 'params': params, 'auc': auc}


best = fmin(
    fn=objective, 
    space=search_space, 
    algo=tpe.suggest,
    max_evals=120, # khảo sát 120 bộ tham số (120 model)
    trials=trials
)

with open('bayes_optimize_results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['dropout', 'depth', 'hidden_size', 'ffn_num_layers', 'test_AUC'])
    for trial in trials.trials:
        p = trial['result']['params']
        writer.writerow([p['dropout'], p['depth'], p['hidden_size'], p['ffn_num_layers'], trial['result']['auc']])

print('Best hyperparameters:', best)
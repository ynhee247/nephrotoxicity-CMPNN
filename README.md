# CMPNN

Source code for our IJCAI 2020 paper [Communicative Representation Learning on Attributed Molecular Graphs](https://www.ijcai.org/Proceedings/2020/0392.pdf)

The code was built based on [DMPNN](https://github.com/chemprop/chemprop). Thanks a lot for their code sharing!




## Dependencies

+ python 3.7
+ cuda >= 8.0
+ cuDNN
+ RDKit
+ torch >= 1.2.0

Tips: Using code `conda install -c rdkit rdkit` can help you install package RDKit quickly.


## Bayesian Optimization

`python bayes_optimize.py --num_folds 5 --epochs 30`


## Training

To run the demo code on dataset CMPNN3_filtered, run:

`python train_demo.py`

To train a model, run:

`python train.py --data_path <path> --dataset_type <type> --num_folds 5 --gpu 0 --epochs 30`

where `<path>` is the path to a CSV file containing a dataset, `<type>` is either "classification" or "regression" depending on the type of the dataset.


## Predicting

`python predict.py --data_path <path> --checkpoint_dir <dir>`

where `<dir>` is the directory where the model checkpoint(s) are saved, and `<path>` is the path of SMILES dataset


## Citation:

Please cite the following paper if you use this code in your work.
```bibtex
@inproceedings{ijcai2020-392,
  title     = {Communicative Representation Learning on Attributed Molecular Graphs},
  author    = {Song, Ying and Zheng, Shuangjia and Niu, Zhangming and Fu, Zhang-hua and Lu, Yutong and Yang, Yuedong},
  booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on
               Artificial Intelligence, {IJCAI-20}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},             
  editor    = {Christian Bessiere}	
  pages     = {2831--2838},
  year      = {2020},
  month     = {7},
  note      = {Main track}
  doi       = {10.24963/ijcai.2020/392},
  url       = {https://doi.org/10.24963/ijcai.2020/392},
}
```

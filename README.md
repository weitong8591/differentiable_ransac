# Fully Differentiable RANSAC Implementation in PyTorch

## Python Environments
Here are some required packages,
```
python = 3.7.11
pytorch = 1.12.1
opencv = 3.4.2
tqdm  
kornia
kornia_moons
tensorboardX = 2.2
scikit-learn
einops
yacs
```
or try with ```conda create --name <env> --file requirements.txt```


[comment]: <> (Example)

[comment]: <> (```)

[comment]: <> ($ conda create --name publish python==3.7.11)

[comment]: <> ($ conda install pytorch=1.12.1)

[comment]: <> ($ conda install pytorch-gpu=1.12.1 cudatoolkit=11.3)

[comment]: <> ($ conda install tqdm)

[comment]: <> ($ conda install tensorboardX)

[comment]: <> ($ conda install sklearn)

[comment]: <> (```)

## Datasets
Saved features can be downloaded from [diff_ransac_data](https://cmp.felk.cvut.cz/~weitong/diff_ransac_data.zip), including Scene St. Peters Square for training, and other 12 scenes for testing.
878M in total, contains two folders: 878M data and 1M evaluation list of numpy files. 
Specify the data path in all of the scripts by parameter '-pth <>'.
RootSIFT feature preparation is referred to [Ransac-tutorial-data](https://github.com/ducha-aiki/ransac-tutorial-2020-data), [NG-RANSAC](https://github.com/vislearn/ngransac).

[comment]: <> (Saved features and models can be downloaded from [here]&#40;https://cmp.felk.cvut.cz/~weitong/&#41;.)

## Implementation
The minimal solvers, model scoring functions and the RANSAC algorithm,local optimization, etc. are re-implemented in PyTorch referring to [MAGSAC](https://github.com/danini/magsac).
Also, thanks to the public repo of [CLNet](https://github.com/sailor-z/CLNet), [NG-RANSAC](https://github.com/vislearn/ngransac), and the libraries of
[PyTorch](https://pytorch.org/get-started/previous-versions/),
[Kornia](https://github.com/kornia/kornia).

## Easy start for E/F matrix estimation test, return AUC scores
```
$ git clone http://github.com/weitong8591/differentiable_ransac.git 
```
test with GPU, RANSAC + local optimization, refer the data path to your own.
```
$ python test.py -nf 2000 -m pretrained_models/saved_model_5PC_l_epi/model.net -bs 32 -fmat 1 -sam 3 -ds sacre_coeur -t 2 -pth <data_path>
``` 
[comment]: <> ([0.5924076, 0.6333666, 0.67357635])
test on a single scene using ```-ds <scene_name>``` , instead, ```-bm 1 ```indicates testing on 12 scenes.
[example_model](pretrained_models/saved_model_5PC_l_epi/model.net) is one of the saved models provided for quick try in this repo, 
feel free to try more models, downloaded from [diff_ransac_models](https://cmp.felk.cvut.cz/~weitong/diff_ransac_models.zip).

train/test with 8PC using```-fmat 1 -sam 3```, 7PC```-fmat 1 -sam 2```, 5PC```-fmat 0 -sam 2```.
Note that we provide this easy start Python testing for simple checking, to reproduce the test results, feel free to go ahead.
## Reproduce the test results
Testing with MAGSAC++ and numerical optimization of essential matrix need C++ pybinding. 
### Python Binding for C++ 
Thanks to the public code of [MAGSAC](https://github.com/danini/magsac.git), we add the proposed Gumbel Softmax Sampler 
inside the C++ implementation, please clone it from [the forked MAGSAC repo including the new sampler](https://github.com/weitong8591/magsac.git), build the project by CMAKE, and compile, install in Python as follows.
```
$ git clone https://github.com/weitong8591/magsac.git --recursive
$ cd magsac
$ mkdir build
$ cd build
$ cmake ..
$ make
$ cd ..
$ python setup.py install
```
### Test with 5PC/7PC for E/F estimation
In C++ MAGSAC, sampler=3 always indicates the Gumbel Softmax Sampler we propose.

```
# test with 5PC for E:
$ python test_magsac.py -nf 2000 -m pretrained_models/saved_model_5PC_l_epi/model.net -bs 32 -fmat 0 -sam 3 -ds sacre_coeur -t 2 -pth <>
```
add ```-fmat 1 ``` to activate fundamental matrix estimation.
## Useful parameters
```
-pth: the source path of all datasets
-sam: choosing samplers, 0 - Uniform sampler, 1,2 - Gumbel Sampler for 5PC/7PC, 3 - Gumbel Sampler for 8PC, default=0
-w0, -w1, -w2: coeffcients of different loss combination, L pose, L classification, L essential
-fmat: 0 - E, 1 - F, default=0
-lr learning rate, default=1e-4
-t: threshold, default=0.75
-e: epoch number, default=10
-bs: batch size of training, default=32
-rbs: batch size of RANSAC iterations, default=64
-tr: train or test model, default=0
-nf: number of features, default=2000
-m: pretrained model or trained m-wodel
-snn: the threshold of SNN ratio filter
-ds dataset name, single dataset
-bm in batch mode, using all the 12 scenes defined in utils.py
-p probabilities, 0-normalized weights, 1-unnormarlized weights, 2-logits, default=2, 
```
## Train the fully differentiable RANSAC
Train the fully differentiable RANSAC end-to-end with the provided initialized weights.

 Using 5PC for E model training, 
```
$ python train.py -nf 2000 -m pretrained_models/weights_init_net_3_sampler_0_epoch_1000_E_rs_r0.80_t0.00_w1_1.00_.net -bs 32 -fmat 0 -sam 2 -tr 1 -w2 1 -t 0.75 -pth <>
```
 8PC for F model training, 
```
$ python train.py -nf 2000 -m pretrained_models/weights_init_net_3_sampler_0_epoch_1000_E_rs_r0.80_t0.00_w1_1.00_.net -bs 32 -fmat 1 -sam 3 -tr 1 -w2 1 -t 0.75 -pth <>
```

## Citation
More details are covered in our paper and feel free to cite it if useful:
```
@article{wei2022fully,
  title={Fully Differentiable RANSAC},
  author={Wei, Tong and Patel, Yash and Matas, Jiri and Barath, Daniel},
  journal={arXiv preprint arXiv:2212.13185},
  year={2022}
}
```
Contact me at weitongln@gmail.com

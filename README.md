# Generalized Differentiable RANSAC (nabla-RANSAC)

>:newspaper: **PyTorch Implementation of the ICCV 2023 paper:**
[Generalized Differentiable RANSAC](https://openaccess.thecvf.com/content/ICCV2023/papers/Wei_Generalized_Differentiable_RANSAC_ICCV_2023_paper.pdf) ($\nabla$-RANSAC).
>
>Tong Wei, Yash Patel, Alexander Shekhovtsov, Jiri Matas and Daniel Barath.

| [paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Wei_Generalized_Differentiable_RANSAC_ICCV_2023_paper.pdf) | [poster](https://cmp.felk.cvut.cz/~weitong/nabla_ransac/poster_nabla_ransac.pdf) | [arxiv](https://arxiv.org/abs/2212.13185https://arxiv.org/abs/2212.1318) | [diff_ransac_models](https://cmp.felk.cvut.cz/~weitong/nabla_ransac/diff_ransac_models.zip) | [diff_ransac_data for E/F](https://cmp.felk.cvut.cz/~weitong/nabla_ransac/diff_ransac_data.zip) | [3d_match_data](https://cmp.felk.cvut.cz/~weitong/nabla_ransac/3d_match_data.zip) | [Ransac-tutorial-data](https://github.com/ducha-aiki/ransac-tutorial-2020-data)

## Updates
<details>
<summary>[2023.10. Solvers in Kornia]</summary>
Our implemented 5PC solver for essential matrix estimation is integrated in [Kornia](https://github.com/kornia/kornia)! Install it from source by

```
$ pip install git+https://github.com/kornia/kornia
```
An example of importing 5PC from [Kornia](https://github.com/kornia/kornia) is shown [here](kornia_5pc_example.ipynb).
</details>

<!-- > :file_folder: **Important links for the trained models and datasets:**
>
>Trained models of 5PC/7PC/8PC for E/F estimation, and 'point_model.net' for 3D point cloud registration are available at [diff_ransac_models](https://cmp.felk.cvut.cz/~weitong/nabla_ransac/diff_ransac_models.zip).
Data for E/F can be downloaded at [diff_ransac_data](https://cmp.felk.cvut.cz/~weitong/nabla_ransac/diff_ransac_data.zip), and [3d_match_data](https://cmp.felk.cvut.cz/~weitong/nabla_ransac/3d_match_data.zip) for point registration. -->
## Implementations and Environments


Here are the required packages,
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

<!-- ## Datasets
Saved features for E/F estimation can be downloaded from [diff_ransac_data](https://cmp.felk.cvut.cz/~weitong/nabla_ransac/diff_ransac_data.zip), including Scene St. Peters Square for training, and other 12 scenes for testing.
878M in total, contains two folders: 878M data and 1M evaluation list of numpy files.
The features of 3DMatch and 3DLoMatch for training, validation and testing can be downloaded from [3d_match_data](https://cmp.felk.cvut.cz/~weitong/nabla_ransac/3d_match_data.zip).

Specify the data path in all of the scripts by parameter '-pth <>'.
RootSIFT feature preparation is referred to [Ransac-tutorial-data](https://github.com/ducha-aiki/ransac-tutorial-2020-data), [NG-RANSAC](https://github.com/vislearn/ngransac).

[comment]: <> (Saved features and models can be downloaded from [here]&#40;https://cmp.felk.cvut.cz/~weitong/&#41;.) -->

## Installation
The SOTA results are tested from our method intergrated in [MAGSAC++](https://github.com/danini/magsac.git), install it in Python as follows.

<!-- Thanks to the public code of [MAGSAC](https://github.com/danini/magsac.git), we add the  -->

<!-- inside the C++ implementation, please clone it from [the forked MAGSAC repo including the new sampler](https://github.com/weitong8591/magsac.git), build the project by CMAKE, and compile, install in Python as follows. -->
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
Note that the proposed Gumbel Softmax Sampler is actiavted by sampler=3. 

## Evaluation
Download the pretrained models we provide [here](https://cmp.felk.cvut.cz/~weitong/nabla_ransac/diff_ransac_models.zip), and test them as follows.

<details>
<summary>[Two-view epipolar geometry estimation]</summary>

Download the RootSIFT features of PhotoTourism from [here](https://cmp.felk.cvut.cz/~weitong/nabla_ransac/diff_ransac_data.zip), and run

```
$ python test_magsac.py -nf 2000 -m pretrained_models/saved_model_5PC_l_epi/model.net -bs 32 -fmat 0 -sam 1 -bm 1 -t 2 -pth <>
```
add ```-fmat 1 ``` to activate fundamental matrix estimation; use ```-ds <scene_name>``` instead of ```-bm 1``` to test on a specific scene.

AUC scores thresholded at [5, 10, 20] are compared for E estimation, F1 scores and median epipolar errors are the evluation metrics for F estimation.

Note: SuperPoint+SuperGlue features on ScanNet are coming soon.
</details>

<details>
<summary>[3D point cloud registration]</summary>

Download the 3DMatch and 3DLoMatch data from [here](https://cmp.felk.cvut.cz/~weitong/nabla_ransac/3d_match_data.zip), and run

```
$ python test_magsac_point.py -m diff_ransac_models/point_model.net -d cpu -us 0 -max 50000 -pth <>
```

The evaluation metrics refer to [registration](registration_utils.py) and [utils](geotransformer/utils/pointcloud.py) in [GeoTransformer](https://arxiv.org/pdf/2202.06688.pdf).


</details>

<details>
<summary>[Learned robust feature matching]</summary>

Download the images, camera intrinsics and extrinsics of PhotoTourism from [here](http://cmp.felk.cvut.cz/~mishkdmy/CVPR-RANSAC-Tutorial-2020/RANSAC-Tutorial-Data-EF.tar), and test with three protocols (-ransac): 0-OpenCV-RANSAC; 0-OpenCV-MAGSAC; 2-MAGSAC++ with PROSAC.

```
$ python test_ransac_loftr.py -nf 2000 -tr 1 -bs 1 -lr 0.000001 -t 3. -sam 3 -fmat 1 -sid loftr -m2 diff_ransac_models/loftr_model.pth -pth <>
```
</details>

## Training

<details>
<summary>[Two-view epipolar geometry estimation]</summary>

Train a importance score prediction model (eg, CLNet backbone, predicting importance score for each input tentative correspondences) with $\nabla$-RANSAC end to end. 

```
$ python train.py -nf 2000 -m pretrained_models/weights_init_net_3_sampler_0_epoch_1000_E_rs_r0.80_t0.00_w1_1.00_.net -bs 32 -fmat 0 -sam 2 -tr 1 -w2 1 -t 0.75 -pth <>
```

Notes: the [initilized weights](pretrained_models/weights_init_net_3_sampler_0_epoch_1000_E_rs_r0.80_t0.00_w1_1.00_.net) are applied; 
5PC is used for essential matrix estimation (-sam 2 -fmat 0); 7PC (-sam 2 -fmat 1) and 8PC (-sam 3 -fmat 1) can be used for F estimation. 

In terms of training loss, -w2 (mean epipolar errors) works the best in terms of AUC scores, however, using the linear combination of the classification loss (-w1 1) with -w2 as objective leads to more normal learning performance (always downward trend, but lower AUC scores in the inference). 
The epipolar loss (w2) is not stable in different training trials.

<!-- ```
$ python train.py -nf 2000 -m pretrained_models/weights_init_net_3_sampler_0_epoch_1000_E_rs_r0.80_t0.00_w1_1.00_.net -bs 32 -fmat 1 -sam 3 -tr 1 -w2 1 -t 0.75 -pth <>
``` -->
</details>

<details>
<summary>[3D point cloud registration]</summary>

<!-- ## :new: Application 2: Apply $\nabla$-RANSAC in 3D point cloud registration -->
Train a importance score prediction model (eg, CLNet backbone, predicting importance score for each input tentative correspondences) with $\nabla$-RANSAC end to end for point cloud registration. Rigid transformation slover is implemented. 

3DMatch train/val dataset is used, 

```
$ python train_point.py -nf 2000 -sam 2 -tr 1 -t 0.75 -pth <>
```
</details>

<details>
<summary>[Learning Robust Feature Matching]</summary>

End-to-end training of feature matcher(eg, LoFTR) with  $\nabla$-RANSAC to improve the predictions of matches and confidences.
Download 'outdoor_ds.ckpt' and put it into [pretrained_models](pretrained_models/).

```
$ python train_ransac_loftr.py -nf 2000 -tr 1 -bs 1 -lr 1e-6 -t 0.75 -sam 3 -fmat 1 -w2 1 -sid loftr -e 50 -p 0 -topk 1 -pth <>
```
</details>


## Demo test

test E estimation on one scene without local optimiztion, no installation of MAGSAC++ needed. 

```
$ python test.py -nf 2000 -m pretrained_models/saved_model_5PC_l_epi/model.net -bs 32 -fmat 1 -sam 3 -ds sacre_coeur -t 2 -pth <data_path>
```
<!-- [comment]: <> ([0.5924076, 0.6333666, 0.67357635])
test on a single scene using ```-ds <scene_name>``` , instead, ```-bm 1 ```indicates testing on 12 scenes.
[example_model](pretrained_models/saved_model_5PC_l_epi/model.net) is one of the saved models for quick try in this repo,
feel free to try more models, [diff_ransac_models](https://cmp.felk.cvut.cz/~weitong/nabla_ransac/diff_ransac_models.zip). -->

<!-- train/test with 8PC using ```-fmat 1 -sam 3```, 7PC ```-fmat 1 -sam 2```, 5PC ```-fmat 0 -sam 2```.
Note that we provide this Python script for simple checking. -->
## Notes

<details>
<summary>[Useful parameters]</summary>

```
-pth: the source path of all datasets
-sam: samplers, 0 - Uniform sampler, 1,2 - Gumbel Sampler for 5PC/7PC, 3 - Gumbel Sampler for 8PC, default=0
-w0, -w1, -w2: coefficients of different loss combination, L pose, L classification, L essential
-fmat: 0 - E, 1 - F, default=0
-lr learning rate, default=1e-4
-t: threshold, default=0.75
-e: epochs, default=10
-bs: batch size, default=32
-rbs: batch size of RANSAC iterations, default=64
-tr: train or test mode, default=0
-nf: number of features, default=2000
-m: pretrained model or trained model
-snn: the threshold of SNN ratio filter
-ds dataset, single dataset
-bm in batch mode, using all the 12 testing scenes defined in utils.py
-p probabilities, 0-normalized weights, 1-unnormarlized weights, 2-logits, default=2,
-topk: whether to get the loss averaged on the topk models or all.
```

</details>

<details>
<summary>[Referred code]</summary>

The minimal solvers, model scoring functions, local optimization, etc. are re-implemented in PyTorch referring to [MAGSAC](https://github.com/danini/magsac).
Also, thanks to the public repo of [CLNet](https://github.com/sailor-z/CLNet), [NG-RANSAC](https://github.com/vislearn/ngransac), and the libraries of
[PyTorch](https://pytorch.org/get-started/previous-versions/),
[Kornia](https://github.com/kornia/kornia).
</details>

## Citation
More details are covered in our paper and feel free to cite it if useful:
```
@InProceedings{wei2023generalized,
  title={Generalized differentiable RANSAC},
  author={Wei, Tong and Patel, Yash and Shekhovtsov, Alexander and Matas, Jiri and Barath, Daniel},
  booktitle={ICCV},
  year={2023}
}
```
Contact me at weitongln@gmail.com

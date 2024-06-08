CUDA配置：
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2017 NVIDIA Corporation
Built on Fri_Sep__1_21:08:03_CDT_2017
Cuda compilation tools, release 9.0, V9.0.176
```
conda配置参照`environment.yml`文件

已经上传运行所需的数据集和脚本，可以使用runxxx脚本开始训练。
`log`文件夹存放了运行日志和参数，`datasets/RADAR`是用到的数据集

以下为运行参数
```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python train_images.py --gammaD {gammaD} --gammaG {gammaG} \
--manualSeed {seed_value} --encoded_noise --preprocessing --cuda --image_embedding {new_SqueezeNet_1024} --class_embedding {new_Video_1024_gzsl} \
--nepoch 300 --ngh 4096 --ndh 4096 --lr {lr} --classifier_lr {lr*10} --lambda1 10 --critic_iter 5 --dataroot datasets --dataset RADAR \
--nclass_all {11} --batch_size 32 --nz 1024 --latent_size 1024 --attSize 1024 --resSize 1024 --syn_num 320 \
--gzsl --recons_weight {recon} --a1 1 --a2 1 --feed_lr 0.00001 --dec_lr 0.0001 --feedback_loop 2
```

- 注意修改{}中的内容后删去{}
- 要同步修改 `--nz --latent_size --attSize` 这三者的值保证与属性的值一致，`resSize`的值与特征的值保持一致
- 通过`--image_embedding` 指定特征，通过`new_Video_1024_gzsl`指定属性
- nclass_all 为全部类型的数目
- 更多参数设置可以参考脚本和config
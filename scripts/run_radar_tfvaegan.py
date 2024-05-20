#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 17:52:45 2019
@author: akshita
"""
import os
# os.system('''CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4  python train_images.py --gammaD 10 --gammaG 1 \
# --manualSeed 3483 --encoded_noise --preprocessing --cuda --image_embedding SqueezeNet_1024 --class_embedding GPT_3072 \
# --nepoch 300 --ngh 4096 --ndh 4096 --lr 0.0005 --classifier_lr 0.005 --lambda1 10 --critic_iter 5 --dataroot datasets --dataset RADAR \
# --nclass_all 27 --batch_size 256 --nz 3072 --latent_size 3072 --attSize 3072 --resSize 1024 --syn_num 320 \
# --recons_weight 0.01 --a1 1 --a2 1 --feed_lr 0.0001 --dec_lr 0.001 --feedback_loop 2''')

# BERT1024
os.system('''CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4  python train_images.py --gammaD 10 --gammaG 1 \
--manualSeed 37 --encoded_noise --preprocessing --cuda --image_embedding SqueezeNet_1024 --class_embedding BERT_1024 \
--nepoch 300 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataroot datasets --dataset RADAR \
--nclass_all 27 --batch_size 256 --nz 1024 --latent_size 1024 --attSize 1024 --resSize 1024 --syn_num 320 \
--recons_weight 0.01 --a1 1 --a2 1 --feed_lr 0.0001 --dec_lr 0.001 --feedback_loop 2''')
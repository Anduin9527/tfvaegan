import os

# 定义要搜索的参数范围
lr_values = [5e-4,1e-4]

seed_values = [3407]        
# seed_values = [3407,37,42]
recon_values = [0.05,0.01]
# 循环遍历参数组合 共16
for lr in lr_values:
    for gammaD in [10,1]:
        for gammaG in [10,1]:
            for seed_value in seed_values:
              for recon in recon_values:
                # 构建日志文件路径
                log_file_path = f"log_1024/Video/re{recon}_lr_{lr}_gammaD_{gammaD}_gammaG_{gammaG}_{seed_value}.log"
                if os.path.exists(log_file_path):
                    print(f"Log file {log_file_path} already exists. Skipping...")
                    continue
                # 构建命令字符串
                command = f'''CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python train_images.py --gammaD {gammaD} --gammaG {gammaG} \
                --manualSeed {seed_value} --encoded_noise --preprocessing --cuda --image_embedding SqueezeNet_1024_T --class_embedding Video_1024_gzsl \
                --nepoch 300 --ngh 4096 --ndh 4096 --lr {lr} --classifier_lr {lr*10} --lambda1 10 --critic_iter 5 --dataroot datasets --dataset RADAR \
                --nclass_all 27 --batch_size 128 --nz 1024 --latent_size 1024 --attSize 1024 --resSize 1024 --syn_num 320 \
                --gzsl --recons_weight {recon} --a1 1 --a2 1 --feed_lr 0.00001 --dec_lr 0.0001 --feedback_loop 2'''
                # 将命令字符串写入日志文件
                os.system(f'echo "{command}" > {log_file_path}')
                # 执行命令并将输出重定向到对应的日志文件中
                print(f"Running command: {command}")
                os.system(f'{command} >> {log_file_path} 2>&1')
                # os.system(f'{command}')
                

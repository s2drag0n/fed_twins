# 实验设置
# 类别均匀度：IID p=0.3 0.7
# 噪声设置：(0.5,0) (1,0.3) (1,0.8)

cd ../..

path_output="experiment/xiaorong/result/"


#nohup python -u main.py --alg fed_twins --dataset cifar10 --model resnet34 --rounds2 450 --lr 0.01 --plr 0.01 --num_users 100 --frac2 0.1 \
#--gpu 0 \
#--level_n_system 1 --level_n_lowerb 0.5 \
#--non_iid_prob_class 0.7 \
#--correction --pre_correction_begin_round 100 --correction_begin_round 200 \
#> $path_output/noise_prior_close.txt 2>&1 &

# iid : (0.5 0.3)(1,0.5) 2
# noniid: p=0.7 (0.5 0.3)(1,0.5) 2

# 无CR=======================================================================================================================================
nohup python -u main.py --alg fed_twins --dataset cifar10 --model resnet34 --rounds2 450 --lr 0.01 --plr 0.01 --num_users 100 --frac2 0.1 \
--gpu 3 \
--level_n_system 0.5 --level_n_lowerb 0.3 \
--iid \
--nocr \
> $path_output/nocr_iid_0.5_0.3_xiaorong.txt 2>&1 &

nohup python -u main.py --alg fed_twins --dataset cifar10 --model resnet34 --rounds2 450 --lr 0.01 --plr 0.01 --num_users 100 --frac2 0.1 \
--gpu 3 \
--level_n_system 1 --level_n_lowerb 0.5 \
--iid \
--nocr \
> $path_output/nocr_iid_1_0.5_xiaorong.txt 2>&1 &

nohup python -u main.py --alg fed_twins --dataset cifar10 --model resnet34 --rounds2 450 --lr 0.01 --plr 0.01 --num_users 100 --frac2 0.1 \
--gpu 3 \
--level_n_system 0.5 --level_n_lowerb 0.3 \
--non_iid_prob_class 0.7 \
--nocr \
> $path_output/nocr_noniid_0.5_0.3_xiaorong.txt 2>&1 &

nohup python -u main.py --alg fed_twins --dataset cifar10 --model resnet34 --rounds2 450 --lr 0.01 --plr 0.01 --num_users 100 --frac2 0.1 \
--gpu 3 \
--level_n_system 1 --level_n_lowerb 0.5 \
--non_iid_prob_class 0.7 \
--nocr \
> $path_output/nocr_noniid_1_0.5_xiaorong.txt 2>&1 &

# 无DR=======================================================================================================================================

nohup python -u main.py --alg fed_twins --dataset cifar10 --model resnet34 --rounds2 450 --lr 0.01 --plr 0.01 --num_users 100 --frac2 0.1 \
--gpu 3 \
--level_n_system 0.5 --level_n_lowerb 0.3 \
--iid \
--nodr \
> $path_output/nodr_iid_0.5_0.3_xiaorong.txt 2>&1 &

nohup python -u main.py --alg fed_twins --dataset cifar10 --model resnet34 --rounds2 450 --lr 0.01 --plr 0.01 --num_users 100 --frac2 0.1 \
--gpu 3 \
--level_n_system 1 --level_n_lowerb 0.5 \
--iid \
--nodr \
> $path_output/nodr_iid_1_0.5_xiaorong.txt 2>&1 &

nohup python -u main.py --alg fed_twins --dataset cifar10 --model resnet34 --rounds2 450 --lr 0.01 --plr 0.01 --num_users 100 --frac2 0.1 \
--gpu 3 \
--level_n_system 0.5 --level_n_lowerb 0.3 \
--non_iid_prob_class 0.7 \
--nodr \
> $path_output/nodr_noniid_0.5_0.3_xiaorong.txt 2>&1 &

nohup python -u main.py --alg fed_twins --dataset cifar10 --model resnet34 --rounds2 450 --lr 0.01 --plr 0.01 --num_users 100 --frac2 0.1 \
--gpu 3 \
--level_n_system 1 --level_n_lowerb 0.5 \
--non_iid_prob_class 0.7 \
--nodr \
> $path_output/nodr_noniid_1_0.5_xiaorong.txt 2>&1 &

## 无双模型 无DR================================================================================================================================
#nohup python -u main.py --alg fed_avg --dataset cifar10 --model resnet34 --rounds2 450 --lr 0.01 --plr 0.01 --num_users 100 --frac2 0.1 \
#--gpu 0 \
#--level_n_system 0.5 --level_n_lowerb 0.3 \
#--iid \
#--fedavgcr \
#> $path_output/fedavgcr_iid_0.5_0.3_xiaorong.txt 2>&1 &
#
#nohup python -u main.py --alg fed_avg --dataset cifar10 --model resnet34 --rounds2 450 --lr 0.01 --plr 0.01 --num_users 100 --frac2 0.1 \
#--gpu 0 \
#--level_n_system 1 --level_n_lowerb 0.5 \
#--iid \
#--fedavgcr \
#> $path_output/fedavgcr_iid_1_0.5_xiaorong.txt 2>&1 &
#
#nohup python -u main.py --alg fed_avg --dataset cifar10 --model resnet34 --rounds2 450 --lr 0.01 --plr 0.01 --num_users 100 --frac2 0.1 \
#--gpu 0 \
#--level_n_system 0.5 --level_n_lowerb 0.3 \
#--non_iid_prob_class 0.7 \
#--fedavgcr \
#> $path_output/fedavgcr_noniid_0.5_0.3_xiaorong.txt 2>&1 &
#
#nohup python -u main.py --alg fed_avg --dataset cifar10 --model resnet34 --rounds2 450 --lr 0.01 --plr 0.01 --num_users 100 --frac2 0.1 \
#--gpu 0 \
#--level_n_system 1 --level_n_lowerb 0.5 \
#--non_iid_prob_class 0.7 \
#--fedavgcr \
#> $path_output/nfedavgcr_noniid_1_0.5_xiaorong.txt 2>&1 &


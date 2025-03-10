# 实验设置
# 类别均匀度：IID p=0.3 0.7
# 噪声设置：(0.5,0) (1,0.3) (1,0.8)

cd ../..

path_output="experiment/noise_prior/result/"

nohup python -u main.py --alg fed_twins --dataset cifar10 --model resnet34 --rounds2 450 --lr 0.01 --plr 0.01 --num_users 100 --frac2 0.1 \
--gpu 0 \
--level_n_system 1 --level_n_lowerb 0.5 \
--non_iid_prob_class 0.7 \
--correction --pre_correction_begin_round 100 --correction_begin_round 200 \
> $path_output/noise_prior_close.txt 2>&1 &

nohup python -u main.py --alg fed_twins --dataset cifar10 --model resnet34 --rounds2 450 --lr 0.01 --plr 0.01 --num_users 100 --frac2 0.1 \
--gpu 0 \
--level_n_system 1 \--level_n_lowerb 0.5 \
--non_iid_prob_class 0.7 \
--correction --pre_correction_begin_round 100 --correction_begin_round 200 \
--noise_prior_open \
> $path_output/noise_prior_open.txt 2>&1 &
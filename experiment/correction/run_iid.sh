# 实验设置
# 类别均匀度：IID p=0.3 0.7
# 噪声设置：(0.5,0) (1,0.3) (1,0.8)

cd ../..

path_output="experiment/loss_on_clean_noise/result/"


# IID 数据分布
nohup python main.py --alg fed --dataset cifar10 --model resnet34 --rounds2 450 --lr 0.01 --plr 0.01 --num_users 100 --frac2 0.1 --gpu 0 --iid --level_n_system 0.5 --level_n_lowerb 0 > $path_output/IID_sys0.5_lower0_descript_lossOnCleanNoise.txt 2>&1 &
nohup python main.py --alg fed_avg_loss_static --dataset cifar10 --model resnet34 --rounds2 450 --lr 0.01 --plr 0.01 --num_users 100 --frac2 0.1 --gpu 0 --iid --level_n_system 1 --level_n_lowerb 0.3 > $path_output/IID_sys1_lower0.3_descript_lossOnCleanNoise.txt 2>&1 &
nohup python main.py --alg fed_avg_loss_static --dataset cifar10 --model resnet34 --rounds2 450 --lr 0.01 --plr 0.01 --num_users 100 --frac2 0.1 --gpu 0 --iid --level_n_system 1 --level_n_lowerb 0.8 > $path_output/IID_sys1_lower0.8_descript_lossOnCleanNoise.txt 2>&1 &

# Non-IID p=0.3 数据分布
nohup python main.py --alg fed_avg_loss_static --dataset cifar10 --model resnet34 --rounds2 450 --lr 0.01 --plr 0.01 --num_users 100 --frac2 0.1 --gpu 0 --level_n_system 0.5 --level_n_lowerb 0 --non_iid_prob_class 0.3 > $path_output/NonIID_p0.3_sys0.5_lower0_descript_lossOnCleanNoise.txt 2>&1 &
nohup python main.py --alg fed_avg_loss_static --dataset cifar10 --model resnet34 --rounds2 450 --lr 0.01 --plr 0.01 --num_users 100 --frac2 0.1 --gpu 0 --level_n_system 1 --level_n_lowerb 0.3 --non_iid_prob_class 0.3 > $path_output/NonIID_p0.3_sys1_lower0.3_descript_lossOnCleanNoise.txt 2>&1 &
nohup python main.py --alg fed_avg_loss_static --dataset cifar10 --model resnet34 --rounds2 450 --lr 0.01 --plr 0.01 --num_users 100 --frac2 0.1 --gpu 2 --level_n_system 1 --level_n_lowerb 0.8 --non_iid_prob_class 0.3 > $path_output/NonIID_p0.3_sys1_lower0.8_descript_lossOnCleanNoise.txt 2>&1 &

# Non-IID p=0.7 数据分布
nohup python main.py --alg fed_avg_loss_static --dataset cifar10 --model resnet34 --rounds2 450 --lr 0.01 --plr 0.01 --num_users 100 --frac2 0.1 --gpu 2 --level_n_system 0.5 --level_n_lowerb 0 --non_iid_prob_class 0.7 > $path_output/NonIID_p0.7_sys0.5_lower0_descript_lossOnCleanNoise.txt 2>&1 &
nohup python main.py --alg fed_avg_loss_static --dataset cifar10 --model resnet34 --rounds2 450 --lr 0.01 --plr 0.01 --num_users 100 --frac2 0.1 --gpu 2 --level_n_system 1 --level_n_lowerb 0.3 --non_iid_prob_class 0.7 > $path_output/NonIID_p0.7_sys1_lower0.3_descript_lossOnCleanNoise.txt 2>&1 &
nohup python main.py --alg fed_avg_loss_static --dataset cifar10 --model resnet34 --rounds2 450 --lr 0.01 --plr 0.01 --num_users 100 --frac2 0.1 --gpu 2 --level_n_system 1 --level_n_lowerb 0.8 --non_iid_prob_class 0.7 > $path_output/NonIID_p0.7_sys1_lower0.8_descript_lossOnCleanNoise.txt 2>&1 &

# 算法：fed_twins

# 数据集:          mnist   cifar10   /clothing1M
# model:          lenet   resnet18  /resnet50
# rounds2:        300     450       /50
# 学习率:          0.1     0.01      /0.001
# 客户端数量:       100     100       /300
# frac2:          0.1     0.1       /0.3
# corr预备:        50      100       /20
# corr开始:        100     200       /35
#level_n_system   1       1
#level_n_lowerb   0.5     0.3

# IID
# --nonIID: p=0.3 p=0.5 p=0.7 p=0.9

# 校正: --correction

# noise_prior: 待定

# gpu:待定

# 总实验数: 2x4=8

# 后续需要加一点指标：每轮噪声修复正确率 错误率
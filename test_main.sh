level_n=0.5
level_b=0.3
gpu_num=0
begin_sel_r=10

python -m torch.utils.bottleneck main.py --alg fed_twins --dataset mnist --model lenet --rounds2 25 --lr 0.1 --plr 0.1 --num_users 100 --frac2 0.1 --begin_sel $begin_sel_r --gpu $gpu_num --level_n_system $level_n --level_n_lowerb $level_b --correction False --correction_begin_round 10 > 1.txt
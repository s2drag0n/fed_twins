import os
import subprocess

# 设置变量
level_n = 1
level_b = 0.5
gpu_num = 0
begin_sel_r = 10
rounds2 = 300
pre_correction_begin_round = 50
correction_begin_round = 100

os.chdir("../")

# 执行Python脚本，并将cProfile的输出重定向到1.txt
profile_cmd = (
    f"python main.py "
    f"--alg fed_twins "
    f"--dataset cifar10 "
    f"--model lenet "
    f"--rounds2 {rounds2} "
    f"--lr 0.1 "
    f"--plr 0.1 "
    f"--num_users 100 "
    f"--frac2 0.1 "
    f"--begin_sel {begin_sel_r} "
    f"--gpu {gpu_num} "
    f"--level_n_system {level_n} "
    f"--level_n_lowerb {level_b} "
    f"--correction "
    f"--pre_correction_begin_round {pre_correction_begin_round} "
    f"--correction_begin_round {correction_begin_round} > experiment/result/cifar_10_1_05.txt 2>&1"
)
subprocess.run(profile_cmd, shell=True)
import argparse
import os
import subprocess

#FedAVG

os.chdir("../../")

# 定义数据集及其参数
listDataset = ["mnist", "cifar10", "cifar100", "clothing1m"]
model_dict = {"mnist": "lenet", "cifar10": "resnet18", "cifar100": "resnet34", "clothing1m": "renet50"}
round_dict = {"mnist": 200, "cifar10": 200, "cifar100": 450, "clothing1m": 50}
client_dict = {"mnist": 100, "cifar10": 100, "cifar100": 50, "clothing1m": 300}
frac2_dict = {"mnist": 0.1, "cifar10": 0.1, "cifar100": 0.2, "clothing1m": 0.3}
lr_dict = {"mnist": 0.1, "cifar10": 0.01, "cifar100": 0.01, "clothing1m": 0.001}
pre_correction_begin_round = {"mnist": 50, "cifar10": 100, "cifar100": 100, "clothing1m": 20}
correction_begin_round = {"mnist": 100, "cifar10": 200, "cifar100": 200, "clothing1m": 35}

# ======================================================================================================================

# 使用 argparse 解析命令行参数
parser = argparse.ArgumentParser(description="Run the training script with specified parameters.")
parser.add_argument("--dataset", type=str, choices=listDataset, required=True, help="Dataset to use (mnist, cifar10, cifar100, clothing1m)")
parser.add_argument("--tau", type=float, choices=[0.1, 0.2, 0.3, 0.4,0.6, 0.7, 0.9], required=True, help="噪声比例")
parser.add_argument("--iid", action="store_true", help="Whether to use IID distribution")
parser.add_argument("--class_sample_ratio", type=float, default=0.7, help="Class sample ratio (e.g., 0.7)")
parser.add_argument("--correction", action="store_true", help="Whether to enable correction")
parser.add_argument("--gpu_num", type=int, default=0, help="GPU number to use (default: 0)")
parser.add_argument("--cl", action="store_true", help="Centralized Learning or Federated Learning")
parser.add_argument("--description", type=str, default="", help="extra description")
parser.add_argument("--algorithm", type=str, default="", help="algorithm")

args = parser.parse_args()

# 设置算法
algorithm = args.algorithm
# 设置训练的数据集
dataset = args.dataset
# 设置异构程度
tau = args.tau
# 设置是否iid
iid = args.iid
# 设置采样比例
class_sample_ratio = args.class_sample_ratio
# 设置是否修正
correction = args.correction
# 设置使用的GPU
gpu_num = args.gpu_num
# 设置描述文字
description = args.description
# 设置是否集中式学习
cl = args.cl

# 方法开始轮数
begin_sel_r = 10

# 设置当前设置的输出路径以及文件名
path_output = "experiment/fl_vs_cl/result/"
filename_output = (
    dataset +
    "_algorithm_" + str(algorithm) +
    "_tau_" + str(tau) +
    "_iid_" + str(iid) +
    "_sample_ratio_" + str(class_sample_ratio) +
    "_cl_" + str(cl) +
    "_description_" + str(description)
)
path = path_output + filename_output

num_users = client_dict[dataset]
frac2 = frac2_dict[dataset]

if cl:
    num_users = 1
    frac2 = 1

# 执行Python脚本，并将cProfile的输出重定向到1.txt
profile_cmd = (
    f"nohup stdbuf -o0 -e0  python main.py "
    f"--alg {algorithm} "

    f"--dataset {dataset} "
    f"--model {model_dict[dataset]} "
    f"--rounds2 {round_dict[dataset]} "
    f"--lr {lr_dict[dataset]} "
    f"--plr {lr_dict[dataset]} "
    f"--num_users {num_users} "
    f"--frac2 {frac2} "
    f"--pre_correction_begin_round {pre_correction_begin_round[dataset]} "
    f"--correction_begin_round {correction_begin_round[dataset]} "

    f"--begin_sel {begin_sel_r} "
    f"--gpu {gpu_num} "

    f"--level_n_system 1 "
    f"--level_n_lowerb {tau} "

    f"--non_iid_prob_class {class_sample_ratio} "
)
# 添加参数
if iid:
    profile_cmd += f"--iid "
if correction:
    profile_cmd += f"--correction "
if cl:
    profile_cmd += f"--cl "

# 添加输出重定向
profile_cmd += f"> {path} 2>&1 &"

subprocess.run(profile_cmd, shell=True)
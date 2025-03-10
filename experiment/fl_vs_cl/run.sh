#!/bin/bash

# 定义 tau 的值
tau_values=(0.1 0.2 0.3 0.4 0.6 0.7 0.9)

# 遍历每个 tau 值并运行实验
for tau in "${tau_values[@]}"; do
    echo "Running fed_avg experiment with tau = $tau"
    nohup python run_with_args.py \
      --algorithm fed_avg \
      --dataset cifar10 \
      --tau "$tau" \
      --gpu_num 1 \
      --description fc_vs_cl &
done

for tau in "${tau_values[@]}"; do
    echo "Running centralized_learning experiment with tau = $tau"
    nohup python run_with_args.py \
      --algorithm centralized_learning \
      --dataset cifar10 \
      --tau "$tau" \
      --gpu_num 2 \
      --description fc_vs_cl \
      --iid \
      --cl &
done

echo "All experiments have been started."




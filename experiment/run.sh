nohup python run_with_args.py \
  --dataset mnist \
  --heterogeneous_level 2 \
  --iid \
  --correction \
  --gpu_num 0 \
  --description UsingKmeans++ &

nohup python run_with_args.py \
  --dataset mnist \
  --heterogeneous_level 2 \
  --iid \
  --gpu_num 0 \
  --description UsingKmeans++ &

nohup python run_with_args.py \
--dataset mnist \
--heterogeneous_level 2 \
--correction \
--gpu_num 1 \
--description UsingKmeans++ &

nohup python run_with_args.py \
--dataset mnist \
--heterogeneous_level 2 \
--gpu_num 1 \
--description UsingKmeans++ &
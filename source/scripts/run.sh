

# convection 

CUDA_VISIBLE_DEVICES=0, python -u main.py --visualize True --system convection --beta 64.0 --gpu --xgrid 512 --nt 200 --N_f 6400 --exp_dir '/cluster/home2/whh/workspace/pinn/exp' --work_dir rep_convection_act_stable --sub_name 'wo_adaptive_slope' --layers 64,64,64,64,64,1 --plot_loss --save_model True --seed 111 --init --adam_lr 2e-3 --epoch 100000 --activation sin --repeat 5 --start_repeat 0 --sample_type grid --N_f_x 64 --N_f_t 100 --fix_sample --linearpool --poolsize '0,1,2,3,4' --aggregate softmax --weight_sharing --coeff_lr 1.0 --coeff_lr_first_layer 1.0 --lr_first_layer 2e-3 --lr_second_layer 2e-3 --tau 1.0 --cosine_decay --warm_up_iter 1000 --sep_cosine_decay --sep_warm_up_iter 1000 --sep_optim --sep_optimizer sgd --momentum 0.75 --l2_reg 0.0 --weight_decay 0.0 --coeff_weight_decay 0.0 --port 39080 --clip 100.0 --disable_lbfgs & 

sleep 10s 

CUDA_VISIBLE_DEVICES=1, python -u main.py --visualize True --system convection --beta 64.0 --gpu --xgrid 512 --nt 200 --N_f 6400 --exp_dir '/cluster/home2/whh/workspace/pinn/exp' --work_dir rep_convection_act_stable --sub_name 'w_adaptive_slope' --layers 64,64,64,64,64,1 --plot_loss --save_model True --seed 111 --init --adam_lr 2e-3 --epoch 100000 --activation sin --repeat 5 --start_repeat 0 --sample_type grid --N_f_x 64 --N_f_t 100 --fix_sample --linearpool --poolsize '0,1,2,3,4' --aggregate softmax --weight_sharing --coeff_lr 1e-3 --coeff_lr_first_layer 1e-3 --lr_first_layer 2e-3 --lr_second_layer 2e-3 --tau 1.0 --cosine_decay --warm_up_iter 1000 --sep_cosine_decay --sep_warm_up_iter 1000 --sep_optim --sep_optimizer adam --coeff_beta1 0.9 --coeff_beta2 0.9 --l2_reg 0.0 --weight_decay 0.0 --coeff_weight_decay 0.0 --port 39080 --clip 100.0 --disable_lbfgs --enable_scaling & 

sleep 10s 


# AC 

CUDA_VISIBLE_DEVICES=2, python -u main.py --visualize True --system AC --beta 0.0 --rho 0.0 --nu 0.0 --data_path './dataset/Allen_Cahn.mat' --exp_dir '/cluster/home2/whh/workspace/pinn/exp' --gpu --xgrid 201 --nt 101 --N_f 8000 --work_dir rep_AC_act_stable_adam --sub_name 'wo_adaptive_slope' --layers 32,32,32,1 --plot_loss --save_model True --seed 111 --adam_lr 1e-3 --init --epoch 40000 --activation sin --sample_type interval --repeat 5 --L_u 0.0 --L_b 0.0 --line_search_fn strong_wolfe --fix_sample --port 29064 --clip 100.0 --linearpool --poolsize '0,1,2,3,4' --aggregate softmax --weight_sharing --coeff_lr 1.6e-2 --coeff_lr_first_layer 1.6e-2 --lr_first_layer 1e-3 --lr_second_layer 1e-3 --cosine_decay --warm_up_iter 1000 --sep_cosine_decay --sep_warm_up_iter 1000 --sep_optim --sep_optimizer adam --coeff_beta1 0.9 --coeff_beta2 0.9 --l2_reg 0.0 --weight_decay 0.0 --coeff_weight_decay 0.0 --hard_ibc & 


sleep 10s 

CUDA_VISIBLE_DEVICES=3, python -u main.py --visualize True --system AC --beta 0.0 --rho 0.0 --nu 0.0 --data_path './dataset/Allen_Cahn.mat' --exp_dir '/cluster/home2/whh/workspace/pinn/exp' --gpu --xgrid 201 --nt 101 --N_f 8000 --work_dir rep_AC_act_stable_adam --sub_name 'w_adaptive_slope' --layers 32,32,32,1 --plot_loss --save_model True --seed 111 --adam_lr 1e-3 --init --epoch 40000 --activation sin --sample_type interval --repeat 5 --L_u 0.0 --L_b 0.0 --line_search_fn strong_wolfe --fix_sample --port 29064 --clip 100.0 --linearpool --poolsize '0,1,2,3,4' --aggregate softmax --weight_sharing --coeff_lr 5e-4 --coeff_lr_first_layer 5e-4 --lr_first_layer 1e-3 --lr_second_layer 1e-3 --cosine_decay --warm_up_iter 1000 --sep_cosine_decay --sep_warm_up_iter 1000 --sep_optim --sep_optimizer adam --coeff_beta1 0.9 --coeff_beta2 0.3 --l2_reg 0.0 --weight_decay 0.0 --coeff_weight_decay 0.0 --hard_ibc --enable_scaling --scaler 1.0 & 

sleep 10s

wait 

# KdV 


CUDA_VISIBLE_DEVICES=0, python -u main.py --visualize True --system KdV --beta 0.0 --rho 0.0 --nu 0.0 --data_path './dataset/KdV.mat' --exp_dir '/cluster/home2/whh/workspace/pinn/exp' --gpu --xgrid 513 --nt 201 --N_f 8000 --work_dir rep_KdV_act_stable_adam_hard_2_ibc --sub_name 'wo_adaptive_slope' --layers 32,32,32,1 --plot_loss --save_model True --seed 111 --adam_lr 1e-3 --init --epoch 40000 --activation sin --sample_type interval --repeat 5 --L_u 0.0 --L_b 1.0 --L_f 1.0 --line_search_fn strong_wolfe --fix_sample --port 29064 --clip 100.0 --extra_N_f 0 --range 0.1 --linearpool --poolsize '0,1,2,3,4' --aggregate softmax --weight_sharing --coeff_lr 1e-3 --coeff_lr_first_layer 1e-3 --lr_first_layer 1e-3 --lr_second_layer 1e-3 --cosine_decay --warm_up_iter 1000 --sep_cosine_decay --sep_warm_up_iter 1000 --sep_optim --sep_optimizer adam --coeff_beta1 0.9 --coeff_beta2 0.9 --l2_reg 0.0 --weight_decay 0.0 --coeff_weight_decay 0.0 --max_iter 15000 --hard_ibc & 

sleep 10s 

CUDA_VISIBLE_DEVICES=1, python -u main.py --visualize True --system KdV --beta 0.0 --rho 0.0 --nu 0.0 --data_path './dataset/KdV.mat' --exp_dir '/cluster/home2/whh/workspace/pinn/exp' --gpu --xgrid 513 --nt 201 --N_f 8000 --work_dir rep_KdV_act_stable_adam_hard_2_ibc --sub_name 'w_adaptive_slope' --layers 32,32,32,1 --plot_loss --save_model True --seed 111 --adam_lr 1e-3 --init --epoch 40000 --activation sin --sample_type interval --repeat 5 --L_u 0.0 --L_b 1.0 --L_f 1.0 --line_search_fn strong_wolfe --fix_sample --port 29064 --clip 100.0 --extra_N_f 0 --range 0.1 --linearpool --poolsize '0,1,2,3,4' --aggregate softmax --weight_sharing --coeff_lr 1e-2 --coeff_lr_first_layer 1e-2 --lr_first_layer 1e-3 --lr_second_layer 1e-3 --cosine_decay --warm_up_iter 1000 --sep_cosine_decay --sep_warm_up_iter 1000 --sep_optim --sep_optimizer adam --coeff_beta1 0.9 --coeff_beta2 0.99 --l2_reg 0.0 --weight_decay 0.0 --coeff_weight_decay 0.0 --max_iter 15000 --hard_ibc --enable_scaling & 

sleep 10s 

# CH 

CUDA_VISIBLE_DEVICES=2, python -u main.py --visualize True --system CH --beta 0.0 --rho 0.0 --nu 0.0 --data_path './dataset/CH_C1_02_2.mat' --exp_dir '/cluster/home2/whh/workspace/pinn/exp' --gpu --xgrid 513 --nt 201 --N_f 12000 --work_dir rep_CH_act_decouple_1_fine_grid_adam_100k --sub_name 'wo_adaptive_slope' --layers 32,32,32,2 --plot_loss --save_model True --seed 111 --adam_lr 1e-3 --init --epoch 100000 --activation sin --sample_type grid --N_f_x 80 --N_f_t 100 --repeat 5 --L_u 100.0 --L_b 1.0 --L_f 1.0 --line_search_fn strong_wolfe --fix_sample --port 29064 --clip 100.0 --linearpool --poolsize '0,1,2,3,4' --aggregate softmax --weight_sharing --coeff_lr 5e-4 --coeff_lr_first_layer 5e-4 --lr_first_layer 1e-3 --lr_second_layer 1e-3 --cosine_decay --warm_up_iter 1000 --sep_cosine_decay --sep_warm_up_iter 1000 --sep_optim --sep_optimizer adam --coeff_beta1 0.9 --coeff_beta2 0.5 --l2_reg 0.0 --weight_decay 0.0 --coeff_weight_decay 0.0 --max_iter 15000 --decouple --four_order --fine_grid & 

sleep 10s 

CUDA_VISIBLE_DEVICES=3, python -u main.py --visualize True --system CH --beta 0.0 --rho 0.0 --nu 0.0 --data_path './dataset/CH_C1_02_2.mat' --exp_dir '/cluster/home2/whh/workspace/pinn/exp' --gpu --xgrid 513 --nt 201 --N_f 12000 --work_dir rep_CH_act_decouple_1_fine_grid_adam_100k --sub_name 'w_adaptive_slope' --layers 32,32,32,2 --plot_loss --save_model True --seed 111 --adam_lr 1e-3 --init --epoch 100000 --activation sin --sample_type grid --N_f_x 80 --N_f_t 100 --repeat 5 --L_u 100.0 --L_b 1.0 --L_f 1.0 --line_search_fn strong_wolfe --fix_sample --port 29064 --clip 100.0 --linearpool --poolsize '0,1,2,3,4' --aggregate softmax --weight_sharing --coeff_lr 1e-3 --coeff_lr_first_layer 1e-3 --lr_first_layer 1e-3 --lr_second_layer 1e-3 --cosine_decay --warm_up_iter 1000 --sep_cosine_decay --sep_warm_up_iter 1000 --sep_optim --sep_optimizer adam --coeff_beta1 0.9 --coeff_beta2 0.9 --l2_reg 0.0 --weight_decay 0.0 --coeff_weight_decay 0.0 --max_iter 15000 --decouple --four_order --fine_grid --enable_scaling & 

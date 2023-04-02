from main import run
import sys
import subprocess
import argparse
from itertools import cycle

parser = argparse.ArgumentParser()
parser.add_argument('--cl_mode', type=str,
                    default='domain',
                    help='The CL paradigm (domain or class).',
                    choices=['domain', 'class'])
parser.add_argument('--visible_gpus', type=str,
                    default="0,1,2,3,4,5,6,7",
                    help='Determines the CUDA devices visible to ' +
                         'the script. A string of comma ' +
                         'separated integers is expected. If the list is ' +
                         'empty, then all GPUs of the machine are used.')
args = parser.parse_args()
cl_mode = args.cl_mode

lr_rec = 40
num_seeds = 5
size = "20,20" if cl_mode == 'domain' else '200,200'
sparsity = '0.4,0.8,0.5' if cl_mode == 'domain' else '0.2,0.8,0'
lr_neg_exponents = [1.0, 1.5, 2.0, 2.5, 3, 3.5, 4, 4.5, 5.0, 5.5] if cl_mode == 'class' else [1.0, 1.5, 2.0, 2.5, 3, 3.5, 4, 4.5, 5.0]
num_classes_per_task = 2
epochs = int((20 // num_classes_per_task) * num_classes_per_task / (10 // num_classes_per_task))


gpus = cycle([int(gpu) for gpu in args.visible_gpus.split(',')])

for seed in range(1, num_seeds + 1):
    
    
    for idx, lr in enumerate([10**(-neg_exp) for neg_exp in lr_neg_exponents]):
        
        dir_name = f'activations/{size=}-{lr=}-{sparsity=}-{lr_rec=}-{num_classes_per_task=}'
        return_codes = []
        # bp
        return_codes.append(subprocess.Popen(f"CUDA_VISIBLE_DEVICES={next(gpus)} python3 run_bp.py --epochs={epochs} --clip_grad_norm=1 --lr={lr} --optimizer=Adam --adam_beta1=0.9 --adam_beta2=0.999 --adam_epsilon=0.002 --dataset=split_mnist --size_hidden={size} --hidden_activation=tanh --initialization=xavier_normal --cl_mode={cl_mode} --random_seed={seed} --record_first_batch_activations --batch_size=512 --num_classes_per_task={num_classes_per_task} --out_dir=out/{dir_name}/bp-{seed=}", shell=True))

        # dfc-sparse-rec
        return_codes.append(subprocess.Popen(f"CUDA_VISIBLE_DEVICES={next(gpus)} python3 run_dfc.py --dataset=split_mnist --hidden_activation=tanh --size_hidden={size} --double_precision --epochs={epochs} --ss --tau_noise=0.05 --tau_f=0.5 --dt_di=0.001 --tmax_di=500 --inst_transmission --time_constant_ratio=0.03553062335953924 --sigma=0.15 --strong_feedback --learning_rule=nonlinear_difference --proactive_controller --cl_mode={cl_mode} --target_class_value=0.9 --clip_grad_norm=1 --initialization=xavier --optimizer=Adam --adam_beta1=0.9 --adam_epsilon=0.00023225688276019436 --batch_size=512 --lr={lr} --use_jacobian_as_fb --error_as_loss_grad --layer_max_sparsities={sparsity} --use_recurrent_weights --rec_learning_neurons=suppressed_only --rec_grad_normalization=incoming --freeze_suppressed_neuron_weights --rec_adaptation=-1 --fw_grad_normalization=incoming --lr_rec={lr_rec} --random_seed={seed} --record_first_batch_activations --num_classes_per_task={num_classes_per_task} --out_dir=out/{dir_name}/dfc-sparse-rec-{seed=}", shell=True))

        # dfc-sparse
        return_codes.append(subprocess.Popen(f"CUDA_VISIBLE_DEVICES={next(gpus)} python3 run_dfc.py --dataset=split_mnist --hidden_activation=tanh --size_hidden={size} --double_precision --epochs={epochs} --ss --tau_noise=0.05 --tau_f=0.5 --dt_di=0.001 --tmax_di=500 --inst_transmission --time_constant_ratio=0.03553062335953924 --sigma=0.15 --strong_feedback --learning_rule=nonlinear_difference --proactive_controller --cl_mode={cl_mode} --target_class_value=0.9 --clip_grad_norm=1 --initialization=xavier --optimizer=Adam --adam_beta1=0.9 --adam_epsilon=0.00023225688276019436 --batch_size=512 --lr={lr} --use_jacobian_as_fb --error_as_loss_grad --layer_max_sparsities={sparsity} --freeze_suppressed_neuron_weights --fw_grad_normalization=incoming --random_seed={seed} --record_first_batch_activations --num_classes_per_task={num_classes_per_task} --out_dir=out/{dir_name}/dfc-sparse-{seed=}", shell=True))

        # dfc
        return_codes.append(subprocess.Popen(f"CUDA_VISIBLE_DEVICES={next(gpus)} python3 run_dfc.py --dataset=split_mnist --hidden_activation=tanh --size_hidden={size} --double_precision --epochs={epochs} --ss --tau_noise=0.05 --tau_f=0.5 --dt_di=0.001 --tmax_di=500 --inst_transmission --time_constant_ratio=0.03553062335953924 --sigma=0.15 --strong_feedback --learning_rule=nonlinear_difference --proactive_controller --cl_mode={cl_mode} --target_class_value=0.9 --clip_grad_norm=1 --initialization=xavier --optimizer=Adam --adam_beta1=0.9 --adam_epsilon=0.00023225688276019436 --batch_size=512 --lr={lr} --use_jacobian_as_fb --error_as_loss_grad --fw_grad_normalization=incoming --random_seed={seed} --record_first_batch_activations --num_classes_per_task={num_classes_per_task} --out_dir=out/{dir_name}/dfc-{seed=}", shell=True))

        # dfc-rec
        return_codes.append(subprocess.Popen(f"CUDA_VISIBLE_DEVICES={next(gpus)} python3 run_dfc.py --dataset=split_mnist --hidden_activation=tanh --size_hidden={size} --double_precision --epochs={epochs} --ss --tau_noise=0.05 --tau_f=0.5 --dt_di=0.001 --tmax_di=500 --inst_transmission --time_constant_ratio=0.03553062335953924 --sigma=0.15 --strong_feedback --learning_rule=nonlinear_difference --proactive_controller --cl_mode={cl_mode} --target_class_value=0.9 --clip_grad_norm=1 --initialization=xavier --optimizer=Adam --adam_beta1=0.9 --adam_epsilon=0.00023225688276019436 --batch_size=512 --lr={lr} --use_jacobian_as_fb --error_as_loss_grad --use_recurrent_weights --rec_learning_neurons=all --rec_grad_normalization=incoming --rec_adaptation=-1 --fw_grad_normalization=incoming --lr_rec={lr_rec} --random_seed={seed} --record_first_batch_activations --num_classes_per_task={num_classes_per_task} --out_dir=out/{dir_name}/dfc-rec-{seed=}", shell=True))

        exit_codes = [p.wait() for p in return_codes]
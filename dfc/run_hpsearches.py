from main import run
import sys
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--visible_gpus', type=str, default="",
                    help='Determines the CUDA devices visible to ' +
                         'the script. A string of comma ' +
                         'separated integers is expected. If the list is ' +
                         'empty, then all GPUs of the machine are used.')
parser.add_argument('--force_dataset', type=str, default="",
                    help='If provided, overrides dataset specified in grid.')
parser.add_argument('--force_permute_labels', action='store_true',
                    help='If provided, sets the permute_labels parameter specified ' +
                         'in the grid to True, overriding the original config setting.')
parser.add_argument('--force_custom_label_permutation', type=str, default=None,
                    help='List of 10 integers representing a custom permutation for the labels.')
parser.add_argument('--force_num_random_seeds', type=int, default=None,
                    help='Number of random seeds, overriding the original config setting.')
parser.add_argument('--force_hidden_activation', type=str, default=None,
                    help='If provided, overrides the hidden_activation arg in the config.')
args = parser.parse_args()


configs = [
    # class-IL
    #'hpconfig_class-mnist-sparse-rec',
    #'hpconfig_class-mnist-sparse-rec-all-variants',
    #'hpconfig_class-split-mnist-bp',
    'hpconfig_class-split-mnist-ewc',
    'hpconfig_class-split-mnist-si',

    #'hpconfig_class-mnist-sparse-rec-min-accu',
    #'hpconfig_class-split-mnist-bp-min-accu',
    'hpconfig_class-split-mnist-ewc-min-accu',
    'hpconfig_class-split-mnist-si-min-accu',

    #'hpconfig_class-mnist-sparse-rec-sparsity-deciding-mix',


    # domain-IL
    'hpconfig_domain-split-mnist-bp',
    #'hpconfig_domain-mnist-sparse-rec',
    #'hpconfig_domain-mnist-sparse-rec-all-variants',
    
    'hpconfig_domain-split-mnist-ewc',
    'hpconfig_domain-split-mnist-si',
    'hpconfig_domain-split-mnist-l2',

    #'hpconfig_domain-mnist-sparse-rec-min-accu',
    'hpconfig_domain-split-mnist-bp-min-accu',
    'hpconfig_domain-split-mnist-ewc-min-accu',
    'hpconfig_domain-split-mnist-si-min-accu',

    #'hpconfig_domain-mnist-sparse-rec-sparsity-deciding-mix',
    #'hpconfig_domain-mnist-sparse-rec-sparsity-search',
    
]

dataset_subdir_string = '.' if args.force_dataset == '' else args.force_dataset
permute_subdir_string = '-permuted' if args.force_permute_labels else ''
hidden_activation_string = f'-{args.force_hidden_activation}' if args.force_hidden_activation else ''
permute_labels_flag = '--force_permute_labels' if args.force_permute_labels else ''
random_seeds_flag = f'--force_num_random_seeds={args.force_num_random_seeds}' if args.force_num_random_seeds else ''
hidden_activation_flag = f'--force_hidden_activation={args.force_hidden_activation}' if args.force_hidden_activation else ''

if args.force_custom_label_permutation:
    permute_subdir_string = f'-custom_label_permutation={args.force_custom_label_permutation}'
    permute_labels_flag = f'--force_custom_label_permutation={args.force_custom_label_permutation}'

for config in configs:
    return_code = subprocess.Popen(f"python3 -m hpsearch.hpsearch --visible_gpus={args.visible_gpus} --force_dataset={args.force_dataset} {permute_labels_flag} {random_seeds_flag} {hidden_activation_flag} --max_num_jobs_per_gpu=1 --allowed_memory=0.3 --grid_module=hpsearch.{config} --force_out_dir --out_dir=out/hpsearches-final/{dataset_subdir_string}{permute_subdir_string}{hidden_activation_string}/{config} --run_cwd=.", shell=True)
    return_code.wait()

#!/usr/bin/env python3
import argparse
import os
import subprocess
import wandb
api = wandb.Api()

first_keys = ['model', 'dataset']
ignore_keys = ['task', 'num_classes', 'num_node_features']


def build_training_command(config: dict):
    out = ['python3 topognn/train_model_gnn_benchmarks.py']
    for key in first_keys:
        out.append(f'--{key}={config[key]}')

    for key, value in config.items():
        if key in first_keys or key in ignore_keys:
            continue
        out.append(f'--{key}={value}')
    return out


def get_run_information(runid: str):
    run = api.run(runid)
    return run


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('runid', type=str)
    args = parser.parse_args()
    run = get_run_information(args.runid)
    training_command = build_training_command(run.config)
    print('Running command:')
    print(' '.join(training_command))
    call_env = os.environ.copy()
    call_env["WANDB_RESUME"] = "must"
    call_env["WANDB_RUN_ID"] = args.runid.split('/')[-1]
    # Run the command
    subprocess.call(' '.join(training_command), shell=True, env=call_env)

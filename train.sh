#!/bin/bash
torchrun --nproc_per_node=8 train_signature_fewshot.py --chisig_dir ./ChiSig --save_dir /dir/to/output_model

export CUDA_VISIBLE_DEVICES=0
python ProSpect_Classifier/finetune_v2_wandb.py --n_samples 50 10\
                                                --to_keep 3 1\

# python ProSpect_Classifier/finetune.py --n_samples 50 10\
#                                         --to_keep 3 1\
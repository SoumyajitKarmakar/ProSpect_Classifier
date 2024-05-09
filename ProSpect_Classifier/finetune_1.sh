# Hyperparameter finetuning for Let it Wag dataset.

CUDA_VISIBLE_DEVICES=1 python ProSpect_Classifier/finetune_v2_wandb.py \
                        --n_samples 50 10 \
                        --to_keep 3 1 \
                        --epoch 200 \
                        --lr 0.005

# CUDA_VISIBLE_DEVICES=1 python ProSpect_Classifier/finetune_v2_wandb.py \
#                         --n_samples 50 10 \
#                         --to_keep 3 1 \
#                         --epoch 200 \
#                         --lr 0.001



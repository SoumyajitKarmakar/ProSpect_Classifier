# export PYTHONPATH="$PYTHONPATH:/data/ankit/ProSpect_main/src/taming-transformers"
# export PYTHONPATH="$PYTHONPATH:/data/ankit/ProSpect_main/src/clip"

export CUDA_VISIBLE_DEVICES=0,1

data_folders=( $(find "/home/soumyajit/Project(-1)/DATASETS/10_worst_liw/" -mindepth 1 -maxdepth 1 -type d -printf '%f\n') )

# data_folders=("${data_folders[@]:10}")
# echo "Folder names after removing the first 10 elements:"

# for folder in "${data_folders[@]}"; do
#     echo "$folder"
# done

run_script() {
    timeout 1.25h python main.py --base configs/stable-diffusion/v1-finetune.yaml \
                   -t \
                   --actual_resume "/home/soumyajit/Project(-1)/Pretrained_weights/sd/sd-v1-4.ckpt" \
                   --gpus 0,1, \
                   --data_root "/home/soumyajit/Project(-1)/DATASETS/10_worst_liw/$1"
}

for folder in "${data_folders[@]}"; do
    run_script "$folder"
done

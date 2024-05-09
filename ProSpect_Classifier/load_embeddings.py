from ProSpect_Classifier.finetune_v1 import (
    load_class_embeddings,
    load_model_from_config,
)
from omegaconf import OmegaConf
import torch


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

config = "configs/stable-diffusion/v1-inference.yaml"
ckpt = "models/sd/sd-v1-4.ckpt"
config = OmegaConf.load(f"{config}")
model = load_model_from_config(config, f"{ckpt}")
model = model.to(device)


class_embeddings = load_class_embeddings(model)

torch.save(class_embeddings, "./ProSpect_Classifier/class_embeddings_actual_try1.pth")

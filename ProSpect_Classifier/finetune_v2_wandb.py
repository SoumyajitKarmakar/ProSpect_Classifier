import wandb
import torch
import argparse
from omegaconf import OmegaConf
from finetune_v1 import (
    load_model_from_config,
    load_img,
    contrastive_loss,
    l2_loss,
    average_error_calcn,
)
import numpy as np
from dataloaders.dataloaders import SetDataManager
import tqdm
from dataloaders.dataloaders_for_classifier import Dataloader_classifier
import os
from sam import SAM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# inference_file = "val_bird_images/train.json"  # Path for training file
# embedding_path = "ProSpect_Classifier/class_embeddings_50_actual.pth"

# inference_file = (
#     "ProSpect_Classifier/data/CUB'/train/train.json"  # Path for training file
# )
# embedding_path = "ProSpect_Classifier/class_embeddings_actual_try1.pth"

inference_file = "/home/soumyajit/Project(-1)/ProSpect_main/novel_let_it_wag/train.json"  # Path for training file
embedding_path = "/home/soumyajit/Project(-1)/ProSpect_main/ProSpect_Classifier/class_embeddings_file/let_it_wag_50_embeddings_ProSpect/let_it_wag_50.pth"

torch.autograd.set_detect_anomaly(True)


def build_optimizer(network, optimizer, learning_rate):
    if optimizer == "sgd":
        optimizer = torch.optim.SGD(network, lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = torch.optim.Adam(network, lr=learning_rate)
    elif optimizer == "sam":
        base_optimizer = torch.optim.SGD
        optimizer = SAM(network, base_optimizer, lr=learning_rate, momentum=0.9)
    return optimizer


def train(model, train_loader, embedding_path, wandb_config=None):
    # wandb.init(
    #     project="l2 + Contrastive Loss",
    #     name="liw_test",
    #     config=wandb_config,
    # )

    # wandb_config = wandb.config

    parser = argparse.ArgumentParser()
    parser.add_argument("--to_keep", nargs="+", type=int, required=True)
    parser.add_argument("--n_samples", nargs="+", type=int, required=True)
    parser.add_argument(
        "--n_trials", type=int, default=1, help="Number of trials per timestep"
    )
    parser.add_argument("--batch_size", "-b", type=int, default=16)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=("float16", "float32"),
        help="Model data type to use",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="l2",
        choices=("l1", "l2", "huber"),
        help="Type of loss to use",
    )

    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--lr", type=float, default=0.005)

    args = parser.parse_args()

    # print(args.epochs)
    # exit()

    assert len(args.to_keep) == len(args.n_samples)

    wandb_config["epochs"] = args.epochs
    wandb_config["learning_rate"] = args.lr

    wandb.init(
        project="Hyperparameter tuning",
        name="liw_" + str(args.epochs) + "_" + str(args.lr),
        config=wandb_config,
    )

    wandb_config = wandb.config

    # print(
    #     "liw_" + str(args.epochs) + "_" + str(args.lr),
    #     wandb_config.epochs,
    #     wandb_config.learning_rate,
    # )
    # exit()

    if train_loader is None:
        dataloader_params = dict(
            image_size=224,
            num_aug=100,
            n_way=5,
            n_support=0,
            n_episode=wandb_config.epochs,
            n_query=15,
        )

        train_loader = SetDataManager(**dataloader_params).get_data_loader(
            inference_file
        )

    ## load embeddings
    class_embeddings = torch.load(embedding_path)

    ## Data loader
    pbar = tqdm.tqdm(train_loader)
    correct = 0
    total = 0

    run_dir = f"./ProSpect_Classifier/class_embeddings_file/{wandb.run.name}"
    os.makedirs(run_dir, exist_ok=True)

    for task_id, (x, x_ft, y) in enumerate(pbar):

        datasets = Dataloader_classifier(x, x_ft, y)
        class_idx = [int((y[i][0]).item()) for i in range(y.shape[0])]
        idxs = list(range(len(datasets)))
        # random.shuffle(idxs)
        idxs_to_eval = idxs
        pbar1 = tqdm.tqdm(idxs_to_eval)
        condition = [class_embeddings[i] for i in class_idx]
        condition = torch.cat(condition, dim=0)
        condition = torch.nn.Parameter(condition)

        optimizer = build_optimizer(
            [condition], wandb_config.optimizer, wandb_config.learning_rate
        )
        condition.cuda()

        if total > 0:
            pbar.set_description(f"Acc: {100 * correct / total:.2f}%")

        running_loss = 0
        running_loss1 = 0
        running_loss2 = 0
        current_correct = 0
        current_total = 0

        for i in pbar1:

            image, x_ft1, label = datasets[i]
            content_image = load_img(image).to(device)

            latent = model.get_first_stage_encoding(
                model.encode_first_stage(content_image)
            )

            loss1 = contrastive_loss(condition, class_idx.index(label))
            loss2 = l2_loss(
                model, latent, args, condition, class_idx.index(label)
            )  # Ankit

            loss = wandb_config.alpha * loss1 + loss2

            running_loss += loss.detach().cpu().item()
            running_loss1 += loss1.detach().cpu().item()
            running_loss2 += loss2.detach().cpu().item()

            if wandb_config.optimizer == "sam":
                loss.backward(retain_graph=True)
                optimizer.first_step(zero_grad=True)

                loss.backward()
                optimizer.second_step(zero_grad=True)

            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            pred_idx, pred_errors = average_error_calcn(model, latent, condition, args)

            if class_idx[pred_idx] == int(label):
                current_correct += 1
            current_total += 1
            if current_total > 0:
                pbar1.set_description(
                    f"loss: {loss2 : .2f} , accuracy : {(current_correct/current_total)*100 :.2f}%"
                )

        if total > 0:
            initial_accuracy = correct / total
        else:
            initial_accuracy = 0

        correct += current_correct
        total += current_total

        avg_loss = running_loss / len(pbar1)
        avg_loss1 = running_loss1 / len(pbar1)
        avg_loss2 = running_loss2 / len(pbar1)
        wandb.log(
            {
                "loss": avg_loss,
                "loss1": avg_loss1,
                "loss2": avg_loss2,
                "epoch": task_id,
                "accuracy": current_correct / current_total,
            }
        )

        for count, i in enumerate(class_idx):
            class_embeddings[i] = condition[count * 10 : count * 10 + 10]

        if initial_accuracy <= (correct / total):
            print(f"Saving the new embeddings ...")
            embedding_path = os.path.join(
                run_dir, f"random_class_embeddings_l2+C_{task_id}.pth"
            )
            torch.save(class_embeddings, f=embedding_path)

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":

    wandb.login()

    parameters_dict = {
        "optimizer": {"values": ["adam"]},  # , 'sgd','adam'
        "learning_rate": {
            # a flat distribution between 0 and 0.1
            "distribution": "uniform",
            "min": 0.001,
            "max": 0.01,
        },
        "alpha": {
            # a flat distribution between 0 and 0.1
            "distribution": "uniform",
            "min": 0.001,
            "max": 0.1,
        },
    }

    metric = {"name": "loss", "goal": "minimize"}
    sweep_config = {"method": "random"}

    sweep_config["metric"] = metric
    sweep_config["parameters"] = parameters_dict

    # sweep_id = wandb.sweep(sweep_config, project="Contrastive_+_l2 Training loss")  ##getting the sweep_id

    ## data_loader
    # dataloader_params = dict(
    #     image_size=224, num_aug=100, n_way=5, n_support=0, n_episode=75, n_query=15
    # )

    # train_loader = SetDataManager(**dataloader_params).get_data_loader(inference_file)

    train_loader = None

    config = "configs/stable-diffusion/v1-inference.yaml"
    ckpt = "models/sd/sd-v1-4.ckpt"
    config = OmegaConf.load(f"{config}")
    model = load_model_from_config(config, f"{ckpt}")
    model = model.to(device)
    wandb_config = {
        "epochs": 75,
        "learning_rate": 0.005,
        # "alpha": 0.01,
        "alpha": 0.0,
        "optimizer": "adam",
        # "optimizer": "sam",
    }

    train(model, train_loader, embedding_path, wandb_config)

    # wandb.agent(sweep_id, lambda: train(model,train_loader, embedding_path), count=1)

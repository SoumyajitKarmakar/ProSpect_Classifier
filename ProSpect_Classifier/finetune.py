from dataloaders.dataloaders import SetDataManager
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import torch
import torch.nn.functional as F
from ldm.models.diffusion.ddim import DDIMSampler
from dataloaders.dataloaders_for_classifier import Dataloader_classifier
from ldm.modules.diffusionmodules.util import make_ddim_timesteps
from ldm.modules.diffusionmodules.openaimodel import UNetModel
from ProSpect_Classifier.scheduler import Scheduler
import os.path as osp
from ProSpect_Classifier.utils import LOG_DIR, get_formatstr
from PIL import Image
import PIL
import numpy as np
import json
import argparse
import tqdm
import datetime
from collections import defaultdict
import random


from einops import rearrange, repeat


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# inference_file = "birds_images/val.json"  ### what inference file we are using
# inference_file = "ProSpect_Classifier/data/CUB'/test/novel.json"  ### what inference file we are using
inference_file = "/home/soumyajit/Project(-1)/ProSpect_main/data_let_it_wag/novel.json"  ### what inference file we are using

json_file = "ProSpect_Classifier/class_embeddings_file/liw_100_0.05/error_dict.json"
log_file = "ProSpect_Classifier/class_embeddings_file/liw_100_0.05/accuracy.txt"


### Loading the model to the config
def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(device)
    model.eval()
    return model


def load_img(img):
    image = img.convert("RGB")
    w, h = image.size
    # print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((512, 512), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def visualize(model, noised_latent, noise_pred, i):
    x_samples = model.decode_first_stage(
        (noised_latent[i]).unsqueeze(0) - (noise_pred[i]).unsqueeze(0)
    )

    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

    for x_sample in x_samples:
        x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), "c h w -> h w c")

        output = Image.fromarray(x_sample.astype(np.uint8))

        output.save(f"test{i}.png")

    return


def contrastive_loss(condition, label_idx):
    ### Here should label index in class_idx should be coming not label iteslf
    n_way = 5
    prospect_stage = 10
    loss = 0

    for i in range(prospect_stage):
        loss_list = torch.exp(
            torch.stack(
                [
                    F.cosine_similarity(
                        condition[label_idx * 10 + i], condition[j * 10 + i]
                    )
                    for j in range(n_way)
                ]
            )
        )
        loss += torch.sum(loss_list)

    loss = torch.log(loss)

    return loss


def average_error_calcn(model, latent, condition, args, all_noise=None, latent_size=64):
    # import pdb; pdb.set_trace()
    T = 1000
    max_n_samples = max(args.n_samples)
    scheduler = Scheduler(beta_start=0.00085, beta_end=0.0120)
    if all_noise is None:
        all_noise = torch.randn(
            (max_n_samples, 4, latent_size, latent_size), device=latent.device
        )
    all_noise = all_noise.half()

    data = dict()
    t_evaluated = set()
    remaining_prmpt_idxs = list(range(len(condition) // 10))
    start = T // max_n_samples // 2

    # conditon = [tensor for list_tensor in condition for tensor in list_tensor]
    t_to_eval = list(range(start, T, T // max_n_samples))[:max_n_samples]

    for n_samples, n_to_keep in zip(args.n_samples, args.to_keep):
        ts = []
        noise_idxs = []
        text_embed_idxs = []
        curr_t_to_eval = t_to_eval[
            len(t_to_eval) // n_samples // 2 :: len(t_to_eval) // n_samples
        ][:n_samples]
        curr_t_to_eval = [t for t in curr_t_to_eval if t not in t_evaluated]
        for prompt_i in remaining_prmpt_idxs:
            for t_idx, t in enumerate(curr_t_to_eval, start=len(t_evaluated)):
                ts.extend([t] * args.n_trials)
                noise_idxs.extend(
                    list(range(args.n_trials * t_idx, args.n_trials * (t_idx + 1)))
                )
                text_embed_idxs.extend([prompt_i] * args.n_trials)

        t_evaluated.update(curr_t_to_eval)

        ### implement the scheduler model here

        # text_embeds = [item for item in condition for _ in range(10//args.n_samples)]   ##

        pred_errors = eval_error(
            model,
            scheduler,
            latent,
            all_noise,
            ts,
            noise_idxs,
            condition,
            text_embed_idxs,
            args.batch_size,
            args.dtype,
            args.loss,
        )

        # match up computed errors to the data
        # import pdb; pdb.set_trace() #Ankit
        for prompt_i in remaining_prmpt_idxs:
            mask = torch.tensor(text_embed_idxs) == prompt_i
            prompt_ts = torch.tensor(ts)[mask]
            prompt_pred_errors = pred_errors[mask]
            if prompt_i not in data:
                data[prompt_i] = dict(t=prompt_ts, pred_errors=prompt_pred_errors)
            else:
                data[prompt_i]["t"] = torch.cat([data[prompt_i]["t"], prompt_ts])
                data[prompt_i]["pred_errors"] = torch.cat(
                    [data[prompt_i]["pred_errors"], prompt_pred_errors]
                )

        # compute the next remaining idxs
        # import pdb; pdb.set_trace() #Ankit
        # weights = torch.exp(torch.tensor([-7 * t for t in t_to_eval]))
        # # weights /= torch.sum(weights)
        # errors = [(-data[prompt_i]['pred_errors']*weights).mean() for prompt_i in remaining_prmpt_idxs]

        errors = [
            -data[prompt_i]["pred_errors"].mean() for prompt_i in remaining_prmpt_idxs
        ]
        best_idxs = torch.topk(
            torch.tensor(errors), k=n_to_keep, dim=0
        ).indices.tolist()
        remaining_prmpt_idxs = [remaining_prmpt_idxs[i] for i in best_idxs]

    # organize the output
    assert len(remaining_prmpt_idxs) == 1
    pred_idx = remaining_prmpt_idxs[0]

    return pred_idx, data


def eval_error(
    model,
    scheduler,
    latent,
    all_noise,
    ts,
    noise_idxs,
    text_embeds,
    text_embed_idxs,
    batch_size=32,
    dtype="float32",
    loss="l2",
):
    # import pdb;pdb.set_trace()      #Ankit
    assert (
        len(ts) == len(noise_idxs) == len(text_embed_idxs)
    )  ### ts is the 10 approximate timesteps (50-950)

    pred_errors = torch.zeros(len(ts), device="cpu")
    idx = 0
    with torch.inference_mode():
        for _ in tqdm.trange(
            len(ts) // batch_size + int(len(ts) % batch_size != 0), leave=False
        ):
            batch_ts = torch.tensor(ts[idx : idx + batch_size])
            noise = all_noise[noise_idxs[idx : idx + batch_size]]

            noised_latent = latent * (scheduler.alphas_cumprod[batch_ts] ** 0.5).view(
                -1, 1, 1, 1
            ).to(device) + noise * (
                (1 - scheduler.alphas_cumprod[batch_ts]) ** 0.5
            ).view(
                -1, 1, 1, 1
            ).to(
                device
            )
            t_input = (
                batch_ts.to(device).half()
                if dtype == "float16"
                else batch_ts.to(device)
            )
            # text_input = text_embeds[idx:idx+batch_size]
            required_idx = [
                (j * 10 + (k // 100)).item()
                for j, k in zip(text_embed_idxs[idx : idx + batch_size], batch_ts)
            ]  # Ankit
            text_input = text_embeds[required_idx]

            noise_pred = model.apply_model(
                noised_latent, t_input, text_input
            )  # shape is (16, 4, 64, 64)

            if loss == "l2":
                error = F.mse_loss(noise, noise_pred, reduction="none").mean(
                    dim=(1, 2, 3)
                )
            elif loss == "l1":
                error = F.l1_loss(noise, noise_pred, reduction="none").mean(
                    dim=(1, 2, 3)
                )
            elif loss == "huber":
                error = F.huber_loss(noise, noise_pred, reduction="none").mean(
                    dim=(1, 2, 3)
                )
            else:
                raise NotImplementedError
            pred_errors[idx : idx + len(batch_ts)] = error.detach().cpu()
            idx += len(batch_ts)
    return pred_errors


def load_class_embeddings(model):
    ## Load the path of embedding manager

    f = open("ProSpect_Classifier/embedding_path.json")
    data = json.load(f)

    ## Prompt of each class loaded

    prompts = ["*"]
    prospect_words = ["*" for _ in range(10)]
    class_embedding = {}

    ## Loading the embedding

    for i, j in zip(data["path"], data["label_idx"]):
        model.cpu()
        model.embedding_manager.load(i)
        model = model.to(device)

        class_embedding[j] = torch.cat(
            model.get_learned_conditioning(prompts, prospect_words=prospect_words),
            dim=0,
        )

    return class_embedding


def main():

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
    parser.add_argument(
        "--n_workers",
        type=int,
        default=1,
        help="Number of workers to split the dataset across",
    )
    parser.add_argument(
        "--worker_idx", type=int, default=0, help="Index of worker to use"
    )
    args = parser.parse_args()
    assert len(args.to_keep) == len(args.n_samples)

    ## Load the model

    config = "configs/stable-diffusion/v1-inference.yaml"
    ckpt = "models/sd/sd-v1-4.ckpt"
    config = OmegaConf.load(f"{config}")
    model = load_model_from_config(config, f"{ckpt}")
    model = model.to(device)

    ## load the data
    dataloader_params = dict(
        image_size=224, num_aug=100, n_way=5, n_support=0, n_episode=300, n_query=15
    )

    novel_loader = SetDataManager(**dataloader_params).get_data_loader(inference_file)

    correct = 0
    total = 0

    pbar = tqdm.tqdm(novel_loader)
    # class_embeddings = load_class_embeddings(model)  #Ankit
    class_embeddings = torch.load(
        # "ProSpect_Classifier/class_embeddings_file/l2+C_7_sam_class/random_class_embeddings_l2+C_24.pth"
        "/home/soumyajit/Project(-1)/ProSpect_main/ProSpect_Classifier/class_embeddings_file/liw_100_0.05/random_class_embeddings_l2+C_99.pth"
    )

    ## Define the optimizers, loss

    for task_id, (x, x_ft, y) in enumerate(pbar):

        datasets = Dataloader_classifier(x, x_ft, y)
        class_idx = [int((y[i][0]).item()) for i in range(y.shape[0])]
        idxs = list(range(len(datasets)))
        random.shuffle(idxs)
        idxs_to_eval = idxs[args.worker_idx :: args.n_workers]
        pbar1 = tqdm.tqdm(idxs_to_eval)
        condition = [class_embeddings[i] for i in class_idx]
        condition = torch.cat(condition, dim=0)
        condition = torch.nn.Parameter(condition)

        # import pdb; pdb.set_trace()     #Ankit
        ## Loading optimizer
        # optimizer = torch.optim.Adam([condition], lr=0.003)

        # condition.cuda()
        # optimizer.cuda()

        if total > 0:
            pbar.set_description(f"Acc: {100 * correct / total:.2f}%")

        error = defaultdict(list)  # Ankit

        # Loading the json file
        try:
            with open(json_file, "r") as file:
                existing_data = json.load(file)
        except FileNotFoundError:
            existing_data = {"predict": [], "actual": []}

        current_total = 0
        current_correct = 0
        running_loss = 0
        for i in pbar1:

            image, x_ft1, label = datasets[i]
            content_image = load_img(image).to(device)

            latent = model.get_first_stage_encoding(
                model.encode_first_stage(content_image)
            )

            pred_idx, pred_errors = average_error_calcn(model, latent, condition, args)
            # loss = contrastive_loss(condition, class_idx.index(label))
            # # loss = (pred_errors[class_idx.index(label)]['pred_errors']).mean()
            # optimizer.zero_grad()
            # loss.backward()

            ## Updating only even gradient
            # with torch.no_grad():
            #     mask = torch.arange(condition.size(0)) % 2 == 0
            #     mask = mask.to(device)
            #     condition.grad *= mask.unsqueeze(-1).unsqueeze(-1)

            # optimizer.step()

            # running_loss += loss.item()

            if class_idx[pred_idx] == int(label):
                current_correct += 1
            current_total += 1
            if current_total > 0:
                pbar1.set_description(
                    f"accuracy : {(current_correct/current_total)*100 :.2f}%"
                )

            print(
                f"the pred idx : {class_idx[pred_idx]}, actual idx : {int(label.item())}"
            )
            error["predict"].append(class_idx[pred_idx])
            error["actual"].append(int(label.item()))

        if total > 0:
            initial_accuracy = correct / total
        else:
            initial_accuracy = 0

        correct += current_correct
        total += current_total
        # for count,i in enumerate(class_idx):
        #     # import pdb;pdb.set_trace()  #Ankit
        #     class_embeddings[i] = torch.cat((condition[count*10:count*10+10][:,:10,:],class_embeddings[i][:,10:,:]),dim=1)

        # if initial_accuracy < (correct/total):
        #     print(f"Saving the new embeddings ...")
        #     torch.save(class_embeddings,f="ProSpect_Classifier/logs/class_embeddings.pth")

        existing_data["predict"].extend(error["predict"])
        existing_data["actual"].extend(error["actual"])

        with open(json_file, "w") as file:
            json.dump(existing_data, file)

        filename = log_file
        with open(filename, "a") as f:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(
                f"The accuracy for the model is {correct/total * 100 : .3f} % , Date and Time: {current_time}, on epoch : {task_id}\n"
            )


if __name__ == "__main__":
    main()

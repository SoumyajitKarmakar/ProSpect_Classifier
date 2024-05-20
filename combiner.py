import os
import json

# json_file = "data_let_it_wag/let_it_wag_50.json"
json_file = "/home/soumyajit/Project(-1)/DATASETS/10_worst_liw/let_it_wag_worst_10.json"
embedding_path = "logs"

x = {}
x["path"] = []
x["label_names"] = []
x["label_idx"] = []

with open(json_file) as f:
    data = json.load(f)

labels_name = {}

for i, j in zip(data["label_idx"], data["label_names"]):
    j = j.replace(" ", "_")
    labels_name[j] = i


for dirs in os.listdir(embedding_path):
    for k in labels_name:
        if dirs.startswith(k):
            path = os.path.join(embedding_path, dirs, "checkpoints")
            for files in os.listdir(path):
                # if files.endswith("5249.pt"):
                if files.endswith("11249.pt"):
                    x["path"].append(os.path.join(path, files))
                    x["label_names"].append(k)
                    x["label_idx"].append(labels_name[k])


with open("ProSpect_Classifier/find_me.json", "w") as outfile:
    # Use json.dump to write the dictionary to the file
    json.dump(x, outfile, indent=4)

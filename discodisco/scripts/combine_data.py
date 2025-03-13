import os
import re
import io
from collections import defaultdict


def read_rels(rels_file):
    lines = io.open(rels_file, "r", encoding='utf-8').read().strip().split("\n")[1:]
    return "\n".join(lines)



def read_conllu(conllu_file):
    content = io.open(conllu_file, "r", encoding='utf-8').read().strip()
    return content


def write2file(data_dict, f_type):
    train_d, dev_d, test_d = [], [], []
    for d in data_dict:
        dataset_d = data_dict[d]
        if "train" in dataset_d:
            train_d.append(dataset_d["train"])
        dev_d.append(dataset_d["dev"])
        test_d.append(dataset_d["test"])

    identifier = "mul.dis.all"
    with io.open(f"{out_dir}/{identifier}/{identifier}_train.{f_type}", "w", encoding="utf-8") as f_train:
        if f_type == "rels":
            f_train.write("\n".join(train_d))
        else:
            f_train.write("\n\n".join(dev_d))

    with io.open(f"{out_dir}/{identifier}/{identifier}_dev.{f_type}", "w", encoding="utf-8") as f_dev:
        if f_type == "rels":
            f_dev.write("\n".join(dev_d))
        else:
            f_dev.write("\n\n".join(dev_d))

    with io.open(f"{out_dir}/{identifier}/{identifier}_test.{f_type}", "w", encoding="utf-8") as f_test:
        if f_type == "rels":
            f_test.write("\n".join(test_d))
        else:
            f_test.write("\n\n".join(test_d))



if __name__ == "__main__":
    # specify paths
    script_dir = os.path.dirname(__file__)
    data_dir = script_dir.replace("scripts", "data_unified")
    out_dir = script_dir.replace("scripts", "output")

    rel_data = defaultdict(defaultdict)
    conllu_data = defaultdict(defaultdict)

    for dataset_dir in os.listdir(data_dir):
        if re.match(r"[a-z]", dataset_dir):
            for file in os.listdir(f"{data_dir}/{dataset_dir}"):
                if file.endswith(".rels"):
                    rel_content = read_rels(f"{data_dir}/{dataset_dir}/{file}")
                    if "train" in file:
                        rel_data[dataset_dir]["train"] = rel_content
                    elif "dev" in file:
                        rel_data[dataset_dir]["dev"] = rel_content
                    elif "test" in file:
                        rel_data[dataset_dir]["test"] = rel_content

                elif file.endswith(".conllu"):
                    conllu_content = read_conllu(f"{data_dir}/{dataset_dir}/{file}")
                    if "train" in file:
                        conllu_data[dataset_dir]["train"] = conllu_content
                    elif "dev" in file:
                        conllu_data[dataset_dir]["dev"] = conllu_content
                    elif "test" in file:
                        conllu_data[dataset_dir]["test"] = conllu_content

    rel_data = dict(sorted(rel_data.items(), key=lambda item: item[0]))
    conllu_data = dict(sorted(conllu_data.items(), key=lambda item: item[0]))

    assert len(rel_data) == len(conllu_data) == 26

    write2file(rel_data, "rels")
    write2file(conllu_data, "conllu")

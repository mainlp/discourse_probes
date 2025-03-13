"""
This script is to take a first step at probing the collected data for discourse relations.
"""

import pandas as pd
import torch
from sklearn.linear_model import SGDClassifier
import re
import os
import sys
from sklearn.model_selection import train_test_split
import random
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_partition(path, partition="train"):
    
    spans, relations = None, None

    load_path = path + "temp/"

    if partition == "test":
        load_path = load_path + "test/"

    # check if a checkpoint exists, if so, load it
    if os.path.exists(load_path + "/spans_encoded_checkpoint.pkl"):
        print("Loading checkpoint...")
    else:
        print("No checkpoint found at", load_path)
        print("Run encode_att.py first.")
        return None, None

    spans = pd.read_pickle(load_path + "/spans_encoded_checkpoint.pkl")
    relations = pd.read_pickle(load_path + "/relations_encoded_checkpoint.pkl")

    intra_avg = torch.load(load_path + "/intra_attention_embedding_avg_checkpoint.pt", weights_only=True)
    intra_max = torch.load(load_path + "/intra_attention_embedding_max_checkpoint.pt", weights_only=True)
    intra_pos = torch.load(load_path + "/intra_attention_embedding_pos_checkpoint.pt", weights_only=True)

    inter_avg = torch.load(load_path + "/inter_attention_embedding_avg_checkpoint.pt", weights_only=True)
    inter_max = torch.load(load_path + "/inter_attention_embedding_max_checkpoint.pt", weights_only=True)
    inter_pos = torch.load(load_path + "/inter_attention_embedding_pos_checkpoint.pt", weights_only=True)

    intra_avg_generator = (v for v in intra_avg)
    spans["intra_attention_embedding_avg"] = spans["computed"].apply(lambda x: next(intra_avg_generator) if x else None)
    intra_max_generator = (v for v in intra_max)
    spans["intra_attention_embedding_max"] = spans["computed"].apply(lambda x: next(intra_max_generator) if x else None)
    intra_pos_generator = (v for v in intra_pos)
    spans["intra_attention_embedding_pos"] = spans["computed"].apply(lambda x: next(intra_pos_generator) if x else None)
    

    inter_avg_generator = (v for v in inter_avg)
    relations["inter_attention_embedding_avg"] = relations["computed"].apply(lambda x: next(inter_avg_generator) if x else None)
    inter_max_generator = (v for v in inter_max)
    relations["inter_attention_embedding_max"] = relations["computed"].apply(lambda x: next(inter_max_generator) if x else None)
    inter_pos_generator = (v for v in inter_pos)
    relations["inter_attention_embedding_pos"] = relations["computed"].apply(lambda x: next(inter_pos_generator) if x else None)

    # new label dicts have case-sensitive keys
    # relations["label"] = relations["label"].map(lambda x: x.lower())

    spans.drop(columns=["computed", "Unnamed: 0", "temp"], inplace=True)
    relations.drop(columns=["computed", "Unnamed: 0", "temp"], inplace=True)

    return spans, relations

def load_checkpoint(path):

    spans, relations = None, None

    spans, relations = load_partition(path, "train")
    spans_test, relations_test = load_partition(path, "test")

    return spans, relations, spans_test, relations_test

def load_checkpoints(path):

    spans, relations = None, None
    spans, relations, spans_test, relations_test = load_checkpoint(path)
    
    # get train dev split
    spans_train = spans[spans["doc_id"].str.find("train") > -1]
    spans_dev = spans[spans["doc_id"].str.find("dev") > -1]

    assert len(spans_train) + len(spans_dev) == len(spans)

    relations_train = relations[relations["doc_id"].str.find("train") > -1]
    relations_dev = relations[relations["doc_id"].str.find("dev") > -1]

    assert len(relations_train) + len(relations_dev) == len(relations)

    return spans_train, relations_train, spans_dev, relations_dev, spans_test, relations_test

def get_datasets(spans_train, relations_train, spans_dev, relations_dev, spans_test, relations_test,
                label_dict,
                framework="all", language="all", 
                include_embs=["inter", "intra1", "intra2"], include_pooling=["max"], verbose=False):


    print("Compiling train set...")
    X_train, y_train, train = compile_dataset(relations_train, spans_train,
                                              language=language, label_dict=label_dict,
                                              include_embs=include_embs,
                                              include_pooling=include_pooling, verbose=verbose, drop_null=True)
    print("Compiling dev set...")
    X_dev, y_dev, dev = compile_dataset(relations_dev, spans_dev,
                                        language=language, label_dict=label_dict,
                                        include_embs=include_embs,
                                        include_pooling=include_pooling, verbose=verbose, drop_null=False)
    

    X_test, y_test, test = None, None, None
    if spans_test is not None and relations_test is not None:
        print("Compiling test set...")
        X_test, y_test, test = compile_dataset(relations_test, spans_test,
                                                language=language, label_dict=label_dict,
                                                include_embs=include_embs,
                                                include_pooling=include_pooling, verbose=verbose, drop_null=False)
    
    return_dict = {"train": (X_train, y_train, train), "dev": (X_dev, y_dev, dev), "test": (X_test, y_test, test)}
    
    languages = ["eng", "nld", "eus", "fas", "por", "spa", "rus", "deu", "ita", "zho", "fra", "tha", "tur"]
    languages_dict = dict((l, l) for l in languages)
    groups = {"germanic": ["eng", "nld", "deu"], "romance": ["por", "spa", "ita", "fra"], "indo-european": ["eng", "nld", "fas", "por", "spa", "rus", "deu", "ita", "fra"], "all": ["eng", "nld", "eus", "fas", "por", "spa", "rus", "deu", "ita", "zho", "fra", "tha", "tur"]}

    all_langs = languages_dict | groups
    print(f"Evaluating on {len(all_langs)} language groups: {all_langs.keys()}")
    
    for lang_group, langs in all_langs.items():
        print(f"Compiling dev set for {lang_group}...")
        X_dev, y_dev, dev = compile_dataset(relations_dev, spans_dev,
                                            language=langs, label_dict=label_dict,
                                            include_embs=include_embs,
                                            include_pooling=include_pooling, verbose=verbose, drop_null=False)
        
        return_dict[f"dev_{lang_group}"] = (X_dev, y_dev, dev)

        print(f"Compiling test set for {lang_group}...")
        X_test, y_test, test = compile_dataset(relations_test, spans_test,
                                                language=langs, label_dict=label_dict,
                                                include_embs=include_embs,
                                                include_pooling=include_pooling, verbose=verbose, drop_null=False)

        return_dict[f"test_{lang_group}"] = (X_test, y_test, test)

    return return_dict

def get_partition(joined, framework, language, label_dict):

    if label_dict is not None:
        joined["label"] = joined["label"].map(label_dict)

    if framework != "all":
        print(f"Filtering for framework: {framework}")
        if type(framework) is str:
            joined = joined[joined["framework"] == framework]
        elif type(framework) is list:
            joined = joined[joined["framework"].isin(framework)]
        else:
            print("Framework must be a string or a list of strings. Got: ", type(framework))

    if language != "all":
        print(f"Filtering for language: {language}")
        if type(language) is str:
            joined = joined[joined["language"] == language]
        elif type(language) is list:
            joined = joined[joined["language"].isin(language)]
        else:
            print("Language must be a string or a list of strings. Got: ", type(language))

    return joined

def replace_missing_by_mean(df, column):
    is_null = df[column].isnull()
    mean = torch.stack(df.loc[~is_null, column].tolist(), dim=0).mean(dim=0)
    df.loc[is_null, column] = [mean] * is_null.sum()
    return df


def compile_dataset(relations, spans,
                    framework="all", language="all", label_dict=None, 
                    include_embs=["inter", "intra1", "intra2"], include_pooling=["max"], 
                    replace_nan=True, gum_uses_orig_label=True, verbose=False, drop_null=False):
    """
    Train probes for the relations classification task.

    Args:
    relations: pd.DataFrame, the relations dataframe
    spans: pd.DataFrame, the spans dataframe
    include_embs: list, the embeddings to include in the probe's featrues(inter span, intra span1, intra span2)
    include_pooling: list, the pooling methods to include in the probe's features (max, avg, pos)
    """

    nspans = len(spans)
    nrels = len(relations)

    # drop not encoded spans/relations
    relations = relations[relations["inter_attention_embedding_avg"].notnull()]
    spans = spans[spans["intra_attention_embedding_avg"].notnull()]

    # column naming is inconsistent, fix it
    relations["orig_label"] = relations["type"]

    if verbose:
        print(f"Number of spans: {len(spans)} ({nspans - len(spans)} dropped)")
        print(f"Number of relations: {len(relations)} ({nrels - len(relations)} dropped)")

    # merge the two dataframes to get dataset

    joined = relations.rename({"unit1": "unit_id"}, axis=1).merge(spans, on=["doc_id", "unit_id"], how="inner", suffixes=("", "_span1")).rename({"unit_id": "unit1_id"}, axis=1)
    joined = joined.rename({"unit2": "unit_id"}, axis=1).merge(spans, on=["doc_id", "unit_id"], how="inner", suffixes=("_span1", "_span2")).rename({"unit_id": "unit2_id"}, axis=1)

    # remove all labels that have less than 100 instances
    # joined = joined[joined["label"].map(joined["label"].value_counts()) > 100]

    if drop_null:
    # drop rows that are misssing embeddings
        before = len(joined)
        joined = joined[joined["inter_attention_embedding_avg"].notnull()]
        joined = joined[joined["intra_attention_embedding_avg_span1"].notnull()]
        joined = joined[joined["intra_attention_embedding_avg_span2"].notnull()]
        after = len(joined)
        if verbose:
            print(f"Number of rows dropped due to missing avg embeddings: {before - after}")

        before = len(joined)
        joined = joined[joined["inter_attention_embedding_max"].notnull()]
        joined = joined[joined["intra_attention_embedding_max_span1"].notnull()]
        joined = joined[joined["intra_attention_embedding_max_span2"].notnull()]
        after = len(joined)
        if verbose:
            print(f"Number of rows dropped due to missing max embeddings: {before - after}")

        before = len(joined)
        joined = joined[joined["inter_attention_embedding_pos"].notnull()]
        joined = joined[joined["intra_attention_embedding_pos_span1"].notnull()]
        joined = joined[joined["intra_attention_embedding_pos_span2"].notnull()]
        after = len(joined)

        if verbose:
            print(f"Number of rows dropped due to missing pos embeddings: {before - after}")

    else:
        # replace missing values with the mean
        joined = replace_missing_by_mean(joined, "inter_attention_embedding_avg")
        joined = replace_missing_by_mean(joined, "intra_attention_embedding_avg_span1")
        joined = replace_missing_by_mean(joined, "intra_attention_embedding_avg_span2")

        joined = replace_missing_by_mean(joined, "inter_attention_embedding_max")
        joined = replace_missing_by_mean(joined, "intra_attention_embedding_max_span1")
        joined = replace_missing_by_mean(joined, "intra_attention_embedding_max_span2")

        joined = replace_missing_by_mean(joined, "inter_attention_embedding_pos")
        joined = replace_missing_by_mean(joined, "intra_attention_embedding_pos_span1")
        joined = replace_missing_by_mean(joined, "intra_attention_embedding_pos_span2")


    joined["language"] = joined["doc_id"].apply(lambda x: re.findall(r".*data/([A-Za-z]+)\.", x)[0])
    joined["framework"] = joined["doc_id"].apply(lambda x: re.findall(r".*data/[A-Za-z]+\.([A-Za-z]+)\.", x)[0])
    joined["task"] = joined["doc_id"].apply(lambda x: re.findall(r".*data/[A-Za-z]+\.[A-Za-z]+\.([A-Za-z]+)", x)[0])

    if gum_uses_orig_label:
        # use orig_label column for the gum dataset
        joined.loc[joined["task"] == "gum", "label"] = joined.loc[joined["task"] == "gum", "orig_label"]

    joined = get_partition(joined, framework, language, label_dict)

    print(f"Number of samples in joined set: {len(joined)}")

    # exclude samples from dev set whose label is not in the train set
    # labels = set(joined["label"].unique())
    
    # TODO: Add this to outer fct
    # dev = dev[dev["label"].apply(lambda x: x in train_labels)]
    # print(f"Number of dev samples dropped due to label not existing in train: {dev_samples - len(dev)}")
    
    if len(joined) == 0:
        print("No samples in joined set after filtering.")
        print("Framework: ", framework)
        print("Language: ", language)
        return None, None, None, None



    X_inter_max_joined = torch.stack(joined["inter_attention_embedding_max"].tolist()).reshape(len(joined), -1)
    X_intra1_max_joined = torch.stack(joined["intra_attention_embedding_max_span1"].tolist()).reshape(len(joined), -1)
    X_intra2_max_joined = torch.stack(joined["intra_attention_embedding_max_span2"].tolist()).reshape(len(joined), -1)

    X_inter_avg_joined = torch.stack(joined["inter_attention_embedding_avg"].tolist()).reshape(len(joined), -1)
    X_intra1_avg_joined = torch.stack(joined["intra_attention_embedding_avg_span1"].tolist()).reshape(len(joined), -1)
    X_intra2_avg_joined = torch.stack(joined["intra_attention_embedding_avg_span2"].tolist()).reshape(len(joined), -1)

    X_inter_pos_joined = torch.stack(joined["inter_attention_embedding_pos"].tolist()).reshape(len(joined), -1)
    X_intra1_pos_joined = torch.stack(joined["intra_attention_embedding_pos_span1"].tolist()).reshape(len(joined), -1)
    X_intra2_pos_joined = torch.stack(joined["intra_attention_embedding_pos_span2"].tolist()).reshape(len(joined), -1)

    inter_joined_list = []
    intra1_joined_list = []
    intra2_joined_list = []

    if "max" in include_pooling:
        inter_joined_list.append(X_inter_max_joined)
        intra1_joined_list.append(X_intra1_max_joined)
        intra2_joined_list.append(X_intra2_max_joined)

    if "avg" in include_pooling:
        inter_joined_list.append(X_inter_avg_joined)
        intra1_joined_list.append(X_intra1_avg_joined)
        intra2_joined_list.append(X_intra2_avg_joined)

    if "pos" in include_pooling:
        inter_joined_list.append(X_inter_pos_joined)
        intra1_joined_list.append(X_intra1_pos_joined)
        intra2_joined_list.append(X_intra2_pos_joined)

    X_inter_joined = torch.cat(inter_joined_list, dim=1)
    X_intra1_joined = torch.cat(intra1_joined_list, dim=1)
    X_intra2_joined = torch.cat(intra2_joined_list, dim=1)

    if replace_nan:
        # replace nan values with zeros
        if verbose:
            print("Replacing NaN values with zeros...")
            # print("Number of NaN values in X_inter_joined: ", torch.isnan(X_inter_joined).sum())
            # print("Number of NaN values in X_intra1_joined: ", torch.isnan(X_intra1_joined).sum())
            print("Number of NaN values in X_intra2_joined: ", torch.isnan(X_intra2_joined).sum())

            # print("Of which are in X_inter_max_joined: ", torch.isnan(X_inter_max_joined).sum())
            # print("Of which are in X_inter_avg_joined: ", torch.isnan(X_inter_avg_joined).sum())
            print("Of which are in X_inter_pos_joined: ", torch.isnan(X_inter_pos_joined).sum(), torch.isnan(X_inter_pos_joined).sum() / torch.prod(torch.tensor(X_inter_pos_joined.shape)))

            # print("Of which are in X_intra1_max_joined: ", torch.isnan(X_intra1_max_joined).sum())
            # print("Of which are in X_intra1_avg_joined: ", torch.isnan(X_intra1_avg_joined).sum())
            print("Of which are in X_intra1_pos_joined: ", torch.isnan(X_intra1_pos_joined).sum(), torch.isnan(X_intra1_pos_joined).sum() / torch.prod(torch.tensor(X_intra1_pos_joined.shape)))

            # print("Of which are in X_intra2_max_joined: ", torch.isnan(X_intra2_max_joined).sum())
            # print("Of which are in X_intra2_avg_joined: ", torch.isnan(X_intra2_avg_joined).sum())
            print("Of which are in X_intra2_pos_joined: ", torch.isnan(X_intra2_pos_joined).sum(), torch.isnan(X_intra2_pos_joined).sum() / torch.prod(torch.tensor(X_intra2_pos_joined.shape)))

        X_inter_joined[torch.isnan(X_inter_joined)] = 0
        X_intra1_joined[torch.isnan(X_intra1_joined)] = 0
        X_intra2_joined[torch.isnan(X_intra2_joined)] = 0

        # replace inf values with ones
        if verbose:
            print("Replacing inf values with ones...")
            # print("Number of inf values in X_inter_joined: ", torch.isinf(X_inter_joined).sum())
            # print("Number of inf values in X_intra1_joined: ", torch.isinf(X_intra1_joined).sum())
            # print("Number of inf values in X_intra2_joined: ", torch.isinf(X_intra2_joined).sum())

        X_inter_joined[torch.isinf(X_inter_joined)] = 1
        X_intra1_joined[torch.isinf(X_intra1_joined)] = 1
        X_intra2_joined[torch.isinf(X_intra2_joined)] = 1

    y_joined = joined["label"].to_numpy()

    if verbose:
        print(f"y_joined shape: {y_joined.shape}")
        print(f"y_joined unique: {len(set(y_joined))}")
        # print a histogram for the classes
        for lab, vc in zip(joined["label"].value_counts().index, joined["label"].value_counts()):
            print(lab,  vc)

    # include span embeddings
    X_joined_list = []

    if "inter" in include_embs:
        X_joined_list.append(X_inter_joined)
    if "intra1" in include_embs:
        X_joined_list.append(X_intra1_joined)
    if "intra2" in include_embs:
        X_joined_list.append(X_intra2_joined)

    X_joined = torch.cat(X_joined_list, dim=1)

    if verbose:
        print(f"X_joined shape: {X_joined.shape}")

    return X_joined, y_joined, joined


def train_probe(X_train, y_train, label_set, X_val=None, y_val=None, batch_size=512, layers=1, lr=0.001, epochs=10, loss="crossentropy", weight_decay=0.01, final_nonlinearity="softmax", results_path=None):
    # TODO: Implement classifier in torch
    
    labl2idx = {lab: i for i, lab in enumerate(label_set)}
    idx2label = {i: lab for lab, i in labl2idx.items()}

    # add unknown label
    if "unknown" not in label_set:
        labl2idx["unknown"] = len(labl2idx)
        idx2label[len(labl2idx)] = "unknown"

    y_train = [labl2idx[lab] for lab in y_train]

    # duplicate any classes with only one sample
    for lab, vc in zip(set(y_train), [y_train.count(lab) for lab in label_set]):
        if vc == 1:
            X_train = torch.cat([X_train, X_train[y_train.index(lab)].unsqueeze(0)], dim=0)
            y_train.append(lab)

    if X_val is None or y_val is None:
        # get validation set
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    else:
        y_val = [labl2idx[lab] for lab in y_val]

    train_dataset = torch.utils.data.TensorDataset(X_train, torch.tensor(y_train))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = torch.utils.data.TensorDataset(X_val, torch.tensor(y_val))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # define the model
    final_layer = torch.nn.Softmax(dim=1)
    if final_nonlinearity == "sigmoid":
        final_layer = torch.nn.Sigmoid()


    if layers <= 1:
        model = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(X_train.shape[1], len(label_set)),
            final_layer
        )
    if layers == 2:
        # influenced by Tenney et. al
        model = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(X_train.shape[1], 512),
            torch.nn.Tanh(),
            torch.nn.LayerNorm(512),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, len(label_set)),
            final_layer
        )

    model.to(device)

    if loss == "crossentropy":
        criterion = torch.nn.CrossEntropyLoss()
    elif loss == "weighted_crossentropy":
        weight = torch.tensor([torch.sum(torch.tensor(y_train) == i) for i in range(len(label_set))]).to(device)
        weight = 1 / (torch.sqrt(weight + 1))
        criterion = torch.nn.CrossEntropyLoss(weight=weight)
    elif loss == "hinge":
        criterion = torch.nn.MultiMarginLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best = None
    best_acc = 0

    if results_path is not None:
        hyperparam_dict = {"batch_size": batch_size, "layers": layers, "lr": lr, "epochs": epochs, "loss": loss, "weight_decay": weight_decay, "final_nonlinearity": final_nonlinearity,
                           "len_label_set": len(label_set), "len_train": len(y_train), "len_val": len(y_val), "label_set": label_set, "model": model.__str__()}
        hyperparam_df = pd.DataFrame({"Hyperparameter": list(hyperparam_dict.keys()), "Value": list(hyperparam_dict.values())})
        hyperparam_df.to_csv(results_path + "hyperparameters.csv")

    for epoch in range(epochs):
        loss_sum = 0
        updates = 0
        for i, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            loss_sum += loss.sum().item()
            updates += batch_size

            if i % (1000 / batch_size) == 0:
                loss_avg = loss_sum / updates
                loss_sum, updates = 0, 0
                print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss_avg}")
                
        with torch.no_grad():
            model.eval()
            total = 0
            correct = 0
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                _, predicted = torch.max(y_pred, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
            
            if correct / total > best_acc:
                best = model.state_dict()
                best_acc = correct / total
                print(f"Epoch: {epoch}, Validation Accuracy: {correct / total} (new best)")
            else:
                print(f"Epoch: {epoch}, Validation Accuracy: {correct / total} (best: {best_acc})")
            model.train()

    model.load_state_dict(best)

    return model, idx2label



def evaluate_probe(classifier, X_dev, y_dev, dev, results_path, idx2label, store_all_results=False):
    # evaluate the model
    if idx2label is None:
        print('This probe was not trained. Likely because of test_only option. Disable if this message is unexpected.')
        return
    label2idx = {lab: i for i, lab in idx2label.items()}
    y_dev = [label2idx[lab] if lab in label2idx else label2idx["unknown"] for lab in y_dev]
    with torch.no_grad():
        classifier.eval()
        X_dev = X_dev.to(device).float()
        y_pred = classifier(X_dev)
        _, predicted = torch.max(y_pred, 1)
        correct = (predicted == torch.tensor(y_dev).to(device)).sum().item()
        score = correct / len(y_dev)

    dev["prediction"] = [idx2label[i.item()] for i in predicted]
    dev["correct"] = dev["label"] == dev["prediction"]

    general_results_stats = dict()
    general_results_stats["accuracy"] = score
    general_results_stats["correct"] = (dev["correct"] == True).sum()
    general_results_stats["incorrect"] = (dev["correct"] == False).sum()
    general_results_stats["dimensions"] = X_dev.shape[1]
    general_results_stats["samples"] = len(y_dev)
    general_results_stats["majority_class"] = dev["label"].value_counts().idxmax()
    general_results_stats["majority_fraction"] = dev["label"].value_counts().max() / len(y_dev)
    general_results_stats["test_acc_corrected"] = (score * len(y_dev) + (23371 - len(y_dev)) * general_results_stats["majority_fraction"]) / 23371
    general_results_stats["dev_acc_corrected"] = (score * len(y_dev) + (25727 - len(y_dev)) * general_results_stats["majority_fraction"]) / 25727

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # print a histogram for languages and accuracy
    lang_acc = dev.groupby("language").agg({"correct": "mean", "doc_id": "count"})
    lang_acc.to_csv(results_path + "lang_acc.csv")
    general_results_stats["overall_lang_acc"] = lang_acc["correct"].mean()

    # print a histogram for labels and accuracy
    lab_acc = dev.groupby("label").agg({"correct": "mean", "doc_id": "count"}).sort_values("doc_id", ascending=False)
    lab_acc.to_csv(results_path + "lab_acc.csv")
    general_results_stats["overall_lab_acc"] = lab_acc["correct"].mean()

    # look at most frequent errors
    most_frequent_errors = dev[~dev["correct"]].groupby(["label", "prediction"]).agg({"correct": "count"}).sort_values("correct", ascending=False)
    most_frequent_errors.to_csv(results_path + "most_frequent_errors.csv")

    # look at dataset accuracy
    dev["dataset"] = dev["doc_id"].apply(lambda x: x.split("/")[3])
    dataset_acc = dev.groupby("dataset").agg({"correct": "mean", "doc_id": "count"})
    dataset_acc.to_csv(results_path + "dataset_acc.csv")

    general_results_stats["overall_dataset_acc"] = dataset_acc["correct"].mean()

    stats_df = pd.DataFrame({"Statistic": list(general_results_stats.keys()), "Value": list(general_results_stats.values())})
    stats_df.to_csv(results_path + "general_stats.csv")

    if store_all_results:
        dev.to_csv(results_path + "test_set_results.csv")

    """
    coefficients = classifier.coef_
    classes = classifier.classes_

    clss = []
    coeff = []
    emb = []
    layer = []
    head = []
    # TODO
    # for each class, get the most important features
    for i, (lab, coef) in enumerate(zip(classes, coefficients)):
        for j, c in enumerate(coef):
            clss.append(lab)
            coeff.append(c)
            emb.append(j // 1024)
            layer.append((j % 1024) // 32)
            head.append((j % 1024) % 32)
            # emb_type = "inter" if j < 1024 else "intra1" if j < 2048 else "intra2"
            # pool 
        
    df_coeffs = pd.DataFrame({"class": clss, "coefficient": coeff, "embedding": emb, "layer": layer, "head": head})
    df_coeffs["embedding"] = df_coeffs["embedding"].map({0: "inter", 1: "intra1", 2: "intra2"})
    df_coeffs["abs_coefficient"] = df_coeffs["coefficient"].abs()

    print()
    print("AVERAGE FEATURE IMPORTANCE")
    avg_feature_importance = df_coeffs.groupby(["embedding", "layer", "head"]).agg({"abs_coefficient": "mean"}).sort_values("abs_coefficient", ascending=False).reset_index()
    print(avg_feature_importance)
    avg_feature_importance.to_csv(results_path + "avg_feature_importance.csv")
    """


def get_label_dicts(data_path="data/"):
    unified_labels = pd.read_json(data_path + "unified_labels.json", orient="index").to_dict()[0]
    label_dicts = {"combined": unified_labels}
    return label_dicts

def probe_and_evaluate(
        spans_train, relations_train, spans_dev, relations_dev, spans_test, relations_test,
        label_dict=None,
        framework="all", language="all",
        results_path="results/",
        only_layers=["all"],
        num_layers=32,
        num_att_heads=32,
        num_embs=3,
        overwrite=False,
        include_embs=["inter", "intra1", "intra2"], 
        include_pooling=["max"],
        verbose=True,
        mlp=True,
        store_all_results=False,
        test_only=False):

    dataset = get_datasets(spans_train, relations_train, spans_dev, relations_dev, spans_test, relations_test,
                            label_dict,
                             framework, language,
                            include_embs=include_embs, include_pooling=include_pooling, verbose=verbose)

    if os.path.exists(results_path):
        if os.path.exists(results_path + "train/general_stats.csv") \
              and os.path.exists(results_path + "dev/general_stats.csv") \
                and os.path.exists("test/general_stats.csv") \
                and not overwrite:
            print("Results path already exists, skipping.")
            return
    else:
        os.makedirs(results_path, exist_ok=True)

    X_train, y_train, train = dataset["train"]
    X_dev, y_dev, dev = dataset["dev"]

    mods = [True, False]
    if mlp is not None:
        mods = [mlp]

    for only_layer in only_layers:
        for mlp in mods:
            if mlp:
                exp_path = results_path + "mlp/"
            else:
                exp_path = results_path + "linear/"

            if type(only_layer) is int:
                exp_path = exp_path + f"layer_{only_layer}/"
                print(f"Training probe for layer {only_layer}...")
                X_train_sub = get_layer_embeddings(X_train, layers=[only_layer], embs=list(range(num_embs)), num_layers=num_layers, num_att_heads=num_att_heads, num_embs=num_embs)
                X_dev_sub = get_layer_embeddings(X_dev, layers=[only_layer], embs=list(range(num_embs)), num_layers=num_layers, num_att_heads=num_att_heads, num_embs=num_embs)
            else:
                exp_path = exp_path + "all_layers/"
                print("Training probe for all layers...")
                X_train_sub = X_train
                X_dev_sub = X_dev

            if not os.path.exists(exp_path):
                os.makedirs(exp_path, exist_ok=True)

            test_exp_path = exp_path + "final_full_train_set_training/"

            if not os.path.exists(test_exp_path):
                os.makedirs(test_exp_path, exist_ok=True)

            print("Train and dev set shape:", X_train.shape, X_dev.shape)

            if y_train is None or len(set(y_train)) < 2:
                print("Not enough samples in the training set.")
                return
        
            label_set = set(y_train).union(set(y_dev))
            if mlp:
                batch_size = 128 if len(y_train) > 100000 else min(64, int(len(y_train) / 5) + 1)
                epochs = int(max(60, batch_size / len(y_train) * 10000)) + 1  # have at least 10000 parameter updates
                if type(only_layer) is int:
                    # layer-wise probes do not need so many epochs
                    epochs = 20
                classifier, idx2label = None, None
                if not test_only:
                    classifier, idx2label = train_probe(X_train_sub, y_train, label_set, batch_size=batch_size, layers=2, lr=0.0001, epochs=epochs, loss="weighted_crossentropy", weight_decay=0.00001, final_nonlinearity="sigmoid", results_path=exp_path)
                test_classifier, test_idx2label = train_probe(X_train_sub, y_train, label_set, X_val=X_dev_sub, y_val=y_dev, batch_size=batch_size, layers=2, lr=0.0001, epochs=epochs, loss="weighted_crossentropy", weight_decay=0.00001, final_nonlinearity="sigmoid", results_path=test_exp_path)
            else:
                batch_size = 64 if len(y_train) > 100000 else min(32, int(len(y_train) / 10) + 1)
                epochs = int(max(60, batch_size / len(y_train) * 10000)) + 1 # have at least 10000 parameter updates
                if type(only_layer) is int:
                    epochs = 20
                classifier, idx2label = None, None
                if not test_only:
                    classifier, idx2label = train_probe(X_train_sub, y_train, label_set, batch_size=batch_size, layers=1, lr=0.0001, epochs=epochs, loss="weighted_crossentropy", weight_decay=0.00001, final_nonlinearity="sigmoid", results_path=exp_path)
                test_classifier, test_idx2label = train_probe(X_train_sub, y_train, label_set, X_val=X_dev_sub, y_val=y_dev, batch_size=batch_size, layers=1, lr=0.0001, epochs=epochs, loss="weighted_crossentropy", weight_decay=0.00001, final_nonlinearity="sigmoid", results_path=test_exp_path)


            for name, (X, y, df) in dataset.items():
               
                if type(only_layer) is int:
                    X = get_layer_embeddings(X, layers=[only_layer], embs=list(range(num_embs)), num_layers=num_layers, num_att_heads=num_att_heads, num_embs=num_embs)

                if "test" in name:
                    path_of_eval = test_exp_path + f"{name}/"
                    evaluate_probe(test_classifier, X, y, df, path_of_eval, test_idx2label, store_all_results=store_all_results)
                
                path_of_eval = exp_path + f"{name}/"
                evaluate_probe(classifier, X, y, df, path_of_eval, idx2label, store_all_results=store_all_results)

            if store_all_results:
                if classifier is not None:
                    weights = classifier.state_dict()
                    torch.save(weights, results_path + f"probe_weights_{mlp}_classifier.pt")
                if test_classifier is not None:
                    weights = test_classifier.state_dict()
                    torch.save(weights, results_path + f"probe_weights_{mlp}_test_classifier.pt")
           


def get_layer_embeddings(X, layers=[0], embs=[0, 1, 2], num_layers=32, num_att_heads=32, num_embs=3):
    """
    Get the embeddings for each layer and attention head.
    """

    if X.shape[1] != num_embs * num_layers * num_att_heads:
        print(f"Expected {num_embs * num_layers * num_att_heads}-dimensional embeddings, got {X.shape[1]}.")
        print(f"Expected {num_embs} embeddings, {num_layers} layers, and {num_att_heads} attention heads.")
        return None

    print(X.shape)
    print("layers", layers, "embs", embs, "num_layers", num_layers, "num_att_heads", num_att_heads, "num_embs", num_embs)

    embs_new = []
    for i in embs:
        # print("Select1:", i)
        for j in layers:
            # print("Select2:", j)
            for k in range(num_att_heads):
                # print("Select3:", k)
                emb = X[:, i * num_layers * num_att_heads + j * num_att_heads + k]
                embs_new.append(emb)
                #print("Add:", emb.shape)
    #print("Stack:", len(embs_new))
    return torch.stack(embs_new, dim=1)

if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("Usage: python run_probes.py <path> <probe-type>")
        sys.exit(1)

    path = sys.argv[1]
    results_path = path + "results/"

    seeds = [14, 42, 999, 5555, 123]

    if path.find("rel_embeddings_") == -1:
        print("Should be a rel_embeddings_ path.")
        sys.exit(1)

    if len(sys.argv) > 2:
        exp_type = sys.argv[2]

    label_set = "combined"
    
    label_dicts = get_label_dicts()
    label_dict = label_dicts[label_set]

    if os.path.exists(path + "temp/inter_attention_embedding_avg_checkpoint.pt"):
        spans, relations, spans_dev, relations_dev, spans_test, relations_test = load_checkpoints(path)
    else:
        print("No checkpoint found, run encode_att.py first.")
        sys.exit(1)

    if not os.path.exists(path + "results"):
        os.makedirs(path + "results")

    # get number of samples per dataset
    relations["dataset"] = relations["doc_id"].apply(lambda x: x.split("/")[3])
    relations_dev["dataset"] = relations_dev["doc_id"].apply(lambda x: x.split("/")[3])
    relations_test["dataset"] = relations_test["doc_id"].apply(lambda x: x.split("/")[3])

    relations["dataset"].value_counts().to_csv(results_path + "train_dataset_counts.csv")
    relations_dev["dataset"].value_counts().to_csv(results_path + "dev_dataset_counts.csv")
    relations_test["dataset"].value_counts().to_csv(results_path + "test_dataset_counts.csv")

    languages = ["all", "eng", "nld", "eus", "fas", "por", "spa", "rus", "deu", "ita", "zho", "fra", "tha", "tur"]
    languages_dict = dict((l, l) for l in languages)
    groups = {"germanic": ["eng", "nld", "deu"], "romance": ["por", "spa", "ita", "fra"], "indo-european": ["eng", "nld", "fas", "por", "spa", "rus", "deu", "ita", "fra"], "all2": ["eng", "nld", "eus", "fas", "por", "spa", "rus", "deu", "ita", "zho", "fra", "tha", "tur"]}

    # combine into one dict
    all_langs = languages_dict | groups

    print(all_langs)

    if exp_type == "unified" or exp_type == "all":

        for seed in seeds:

            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            # run framework-specific probes
            for group_name, langs in all_langs.items():

                print(group_name, langs)
            
                label_dict = label_dicts["combined"]
                
                exp_path = results_path + f"seed={seed}/unified/train_lang={group_name}/"

                probe_and_evaluate(spans, relations, spans_dev, relations_dev, spans_test, relations_test,
                                language=langs,
                                label_dict=label_dict,
                                results_path=exp_path,
                                overwrite=True,
                                test_only=True)


    if (exp_type == "layer-wise" or exp_type == "all") and path.find("aya23") != -1:

        for seed in seeds:

            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        
            # run layer-wise probes
            # infer the number of layers and attention heads from the embeddings
            shape = relations["inter_attention_embedding_max"].iloc[0].shape
            print(shape)
            
            model_params = {"aya23_35b": {"layers": 40, "att_heads": 64},
                            "aya23_8b": {"layers": 32, "att_heads": 32},}
            
            print(path)
            model_name = path.split("rel_embeddings_")[1].split("/")[0]
            print(model_name)

            num_layers = model_params[model_name]["layers"]
            num_att_heads = model_params[model_name]["att_heads"]

            label_dict = label_dicts["combined"]


            for pooling_type in [["max"]]: #, ["avg"], ["pos"]]:
                    
                num_embs=len(pooling_type)*3
                if "pos" in pooling_type:
                    num_embs += 3 

                exp_path = results_path + f"seed={seed}/layer-wise/pooling={"_".join(pooling_type)}/"
                probe_and_evaluate(
                    spans, relations, spans_dev, relations_dev, spans_test, relations_test,
                    label_dict=label_dict,
                    results_path=exp_path,
                    only_layers=list(range(num_layers)),
                    num_layers=num_layers,
                    num_att_heads=num_att_heads,
                    num_embs=num_embs,
                    include_embs=["inter", "intra1", "intra2"],
                    include_pooling=pooling_type,
                    overwrite=True,
                    test_only=True)

    if (exp_type == "store-full-preds" or exp_type=="all") and path.find("aya23") != -1:

        for seed in seeds:

            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)


            label_dict = label_dicts["combined"]
            langs = "all"
            exp_path = results_path + f"seed={seed}/store_full_preds/"
            
            probe_and_evaluate(spans, relations, spans_dev, relations_dev, spans_test, relations_test,
                        language=langs,
                        label_dict=label_dict,
                        results_path=exp_path,
                        overwrite=True,
                        store_all_results=True,
                        test_only=True)
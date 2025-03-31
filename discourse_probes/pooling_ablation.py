"""
This script implements the ablation study for the pooling strategies.
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
from run_probes import get_label_dicts, load_checkpoints, probe_and_evaluate

import threading

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python run_probes.py <path> <probe-type>")  
        sys.exit(1)

    path = sys.argv[1]
    # path = "data/disrpt_private/rel_embeddings_aya23_35b/"
    results_path = path + "results/"
    # probe_type = "all"
    probe_type = sys.argv[2]

    if path.find("rel_embeddings_") == -1:
        print("Should be a rel_embeddings_ path.")
        sys.exit(1)

    seeds = [14, 42, 999, 5555, 123]
    label_dict = pd.read_json("data/unified_labels.json", orient="index").to_dict()[0]

    if os.path.exists(path + "temp/inter_attention_embedding_avg_checkpoint.pt"):
        spans, relations, spans_dev, relations_dev, spans_test, relations_test = load_checkpoints(path)
    else:
        print("No checkpoint found, run encode_att.py first.")
        # sys.exit(1)


    relations["dataset"] = relations["doc_id"].apply(lambda x: x.split("/")[3])
    relations_dev["dataset"] = relations_dev["doc_id"].apply(lambda x: x.split("/")[3])
    relations_test["dataset"] = relations_test["doc_id"].apply(lambda x: x.split("/")[3])

    # Add dataset columns

    languages = ["all", "eng", "nld", "eus", "fas", "por", "spa", "rus", "deu", "ita", "zho", "fra", "tha", "tur"]
    languages_dict = dict((l, l) for l in languages)
    groups = {"germanic": ["eng", "nld", "deu"], "romance": ["por", "spa", "ita", "fra"], "indo-european": ["eng", "nld", "fas", "por", "spa", "rus", "deu", "ita", "fra"], "all2": ["eng", "nld", "eus", "fas", "por", "spa", "rus", "deu", "ita", "zho", "fra", "tha", "tur"]}

    # combine into one dict
    all_langs = languages_dict | groups

    print(all_langs)

    embs_ablations = [["inter"], ["intra1", "intra2"], ["intra1", "intra2", "inter"]]
    pooling_ablations = [["avg"], ["max"], ["avg", "max"]]

    threads = []

    if probe_type == "all" or probe_type == "overall_performance":
        # measure overall probe performance

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            for emb_ablation in embs_ablations:
                for pooling_ablation in pooling_ablations:
                    print("Running probe for seed", seed, "emb_ablation", emb_ablation, "pooling_ablation", pooling_ablation)

                    # thread_i = threading.Thread(target=probe_and_evaluate, 
                    #                             args=(spans, relations, spans_dev, relations_dev, spans_test, relations_test),
                    #                             kwargs={"language": "all",
                    #                                     "label_dict": label_dict,
                    #                                     "results_path": results_path + f"pooling_ablations/overall_performance/seed={seed}/embs={','.join(emb_ablation)}/pooling={','.join(pooling_ablation)}/",
                    #                                     "overwrite": True,
                    #                                     "test_only": False,
                    #                                     "include_embs": emb_ablation,
                    #                                     "include_pooling": pooling_ablation})
        
                    probe_and_evaluate(spans, relations, spans_dev, relations_dev, spans_test, relations_test,
                                        language="all",
                                        label_dict=label_dict,
                                        results_path=results_path + f"pooling_ablations/overall_performance/seed={seed}/embs={','.join(emb_ablation)}/pooling={','.join(pooling_ablation)}/",
                                        overwrite=True,
                                        test_only=False,
                                        include_embs=emb_ablation,
                                        include_pooling=pooling_ablation)
                    

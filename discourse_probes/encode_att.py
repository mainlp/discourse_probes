"""
Encode translation prompts into the hidden representations of a language model.
"""

from encode import encode_all_docs, load_data, load_checkpoint, get_model, TOCPU
import encode
import sys
import os

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage: python encode_att.py <path> <model_name>")
        sys.exit(1)

    path = sys.argv[1]
    model_name = sys.argv[2]

    if len(sys.argv) > 3:
        encode.TOCPU = sys.argv[3] == "cpu"
        TOCPU = encode.TOCPU
        print("TOCPU:", TOCPU)

    if not os.path.exists(path):
        print("Creating directory:", path)
        os.makedirs(path)
    
    if not os.path.exists(path + "spans_train.csv"):
        print("Copying rel_embeddings/* to", path)
        os.system(f"cp -r {path}/../rel_embeddings/* " + path)

    docs, spans, relations = load_data(path, "train")
    docs_test, spans_test, relations_test = load_data(path, "test")
    
    print(docs.head())
    print(spans.head())
    print(spans["char_indices"][0])
    print(type(spans["char_indices"][0]))
    print(type(spans["char_indices"][0][0]))
    print(relations.head())

    print("Total docs:", len(docs))
    print("Total spans:", len(spans))
    print("Total relations:", len(relations))

    print("Total docs (test):", len(docs_test))
    print("Total spans (test):", len(spans_test))
    print("Total relations (test):", len(relations_test))

    docs["task_id"] = docs["doc_id"].apply(lambda x: "_".join(x.split("/")[3:5]))

    print(docs.groupby("task_id").size())

    print("Total test gum:", sum(docs_test["doc_id"].str.find(".gum") > 0))

    tokenizer, model = get_model(model_name)

    spans, relations = load_checkpoint(path, spans, relations)

    print(spans.head())
    print(relations.head())

    print(model_name)
    print("32b" in model_name)
    print("35B" in model_name)
    max_length = 3800 if "32b" in model_name else 4000
    max_length = 3800 if "32B" in model_name else max_length
    max_length = 3400 if "35B" in model_name else max_length
    max_length = 3400 if "70B" in model_name else max_length
    max_length = 3400 if "72B" in model_name else max_length
    print("Max length:", max_length)

    #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    spans, relations = encode_all_docs(docs, spans, relations, tokenizer, model, checkpoint_path=path+"temp/",
                                            min_length=0, max_length=max_length)
    
    # print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30))
    # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))

    # spans.to_pickle(path + "spans_train_encoded.pkl")
    # relations.to_pickle(path + "relations_train_encoded.pkl")

    spans_test, relations_test = encode_all_docs(docs_test, spans_test, relations_test, tokenizer, model, checkpoint_path=path+"temp/test/", max_length=max_length)

    spans_test.to_pickle(path + "spans_test_encoded.pkl")
    relations_test.to_pickle(path + "relations_test_encoded.pkl")

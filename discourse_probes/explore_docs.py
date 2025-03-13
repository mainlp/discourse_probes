from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import re
import warnings
import os

warnings.filterwarnings("ignore")


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

def get_model(name = "MaLA-LM/emma-500-llama2-7b", device=device):

    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, device_map="auto")
    # model = nn.DataParallel(model, device_ids=[0, 1])
    # model.to(device)

    return tokenizer, model

def load_data(path, partition):
    docs = pd.read_csv(path + f"documents_{partition}.csv")
    spans = pd.read_pickle(path + "temp/spans_encoded_checkpoint.pkl")
    relations = pd.read_pickle(path + "temp/relations_encoded_checkpoint.pkl")

    # convert char_indices to list of ints
    # spans["char_indices"] = spans["char_indices"].map(lambda x: [int(i) for i in re.findall(r"\d+", x)])

    return docs, spans, relations

tokenizer, model = get_model()

path = "data/disrpt_private/rel_embeddings/"
partition = "train"
docs, spans, relations = load_data(path, partition)

print(f"Number of documents: {len(docs)}")
print(f"Number of spans: {len(spans)}")
print(f"Number of relations: {len(relations)}")

print(f"Number of encoded spans: {len(spans[spans['intra_attention_embedding'].notnull()])}")

for i, row in docs.iterrows():
    tokens = tokenizer(row["document"], return_tensors="pt")
    if len(tokens["input_ids"][0]) > 8000 and not row["doc_id"].find("tha") > -1:
        print(f"Document {row['doc_id']} has {len(tokens['input_ids'][0])} tokens:")
        print(f"{row["document"][0:100]}[...]  (#chars: {len(row["document"])})")
        print("\n\n")

# store embeddings separately from the rest of the data
path = "data/disrpt_private/rel_embeddings/tensors/"
os.makedirs(path, exist_ok=True)

encoded = spans[spans["intra_attention_embedding"].notnull()]

matrices = [torch.stack(emb) for emb in encoded["intra_attention_embedding"]]

spans["encoded"] = spans["intra_attention_embedding"].notnull()

# drop the embeddings from the dataframe
spans.drop(columns=["intra_attention_embedding", "temp"], inplace=True)

# save the tensors and the dataframe
torch.save(matrices, path + "encoded_tensors_train.pt")
spans.to_pickle(path + "spans_encoded_train.pkl")

# can we recombine the tensors?
matrices_test = torch.load(path + "encoded_tensors_train.pt")
print(len(matrices_test))

# recombine the tensors
spans_test = pd.read_pickle(path + "spans_encoded_train.pkl")

# add the embeddings back to the dataframe
spans_test["intra_attention_embedding"] = None
# add embeddings to the entries with "encoded" flag
new_col = []
ct = 0
for i, row in spans_test.iterrows():
    if row["encoded"]:
        new_col.append(matrices_test[ct])
        ct += 1
    else:
        new_col.append(None)

spans_test["intra_attention_embedding"] = new_col


# do same for relations
relations["encoded"] = relations["inter_attention_embedding"].notnull()
encoded = relations[relations["encoded"]]

matrices = [torch.stack(emb) for emb in encoded["inter_attention_embedding"]]
relations.drop(columns=["inter_attention_embedding", "temp"], inplace=True)

torch.save(matrices, path + "encoded_tensors_relations_train.pt")
relations.to_pickle(path + "relations_encoded_train.pkl")
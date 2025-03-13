"""
Encode translation prompts into the hidden representations of a language model.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch
import pandas as pd
import re
import warnings
import os
import time

from torch.profiler import profile, record_function, ProfilerActivity

warnings.filterwarnings("ignore")

TOCPU = False


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"Number of GPUs: {torch.cuda.device_count()}")


def get_model(name="MaLA-LM/emma-500-llama2-7b", device=device, to_cpu=TOCPU):

    device_map = "balanced_low_0"
    if to_cpu or True:
        device_map = "balanced_low_0"
    
    # if name.lower().find("qwen") > 0 or True:
    #    device_map = "auto"

    tokenizer = AutoTokenizer.from_pretrained(name)
    if name == "CohereForAI/aya-101" or name.lower() == "cohereforai/aya-101":
        model = AutoModelForSeq2SeqLM.from_pretrained(name, device_map=device_map, attn_implementation="eager")
    elif name == "meta-llama/Meta-Llama-3-8B":
        model = AutoModelForCausalLM.from_pretrained(name, device_map=device_map, attn_implementation="eager")
    else:
        model = AutoModelForCausalLM.from_pretrained(name, device_map=device_map, attn_implementation="eager", torch_dtype="auto")
    # model = nn.DataParallel(model, device_ids=[0, 1])
    # model.to(device)


    return tokenizer, model


def get_outputs(model, tokens):
    # tokens["decoder_input_ids"] = tokens["input_ids"]

    torch.cuda.empty_cache()

    # print(help(model))

    with torch.no_grad():
        output = model(**tokens, output_hidden_states=False, output_attentions=True, use_cache=False)

    # last_states = [a[:, -1, :].detach().cpu() for a in output.hidden_states]

    output = output.attentions
    
    global TOCPU
    if TOCPU and False: # or len(tokens["input_ids"]) >= 3400:
        # print("Moving output to CPU...")
        output = [a.cpu() for a in output]
    
    # put to cpu -> disabled for now, use a.cpu()
    else:
        output = [a for a in output]
    
    # for i, att in enumerate(output):
    #     if att.dtype == torch.bfloat16:
    #         output[i] = att.float()

    # torch.cuda.empty_cache()

    return None, output, tokens

def add_span_masks(tokens, document, spans, tokenizer=None):
    """
    For a tokenization result, get the token ids of the tokens covering the span.

    Args:
        tokens (list): Tokenization result from tokenizer. If this is not a list of strings, 
            it will be converted to one through the tokenizer's decode method.
        text (str): Original text.
        span_boundaries (list): List of tuples with int span boundaries.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer object. Optional.
    """
    #if tokens[0] is not str:
    #    tokens = [tokenizer.decode(t) for t in tokens]
    
    # align tokens with text
    token_spans = []
    offset = 0

    current_token = []

    for token in tokens:
        token_decoded = tokenizer.decode(token)
        if tokenizer is not None and (token_decoded == tokenizer.pad_token or token_decoded == tokenizer.eos_token or
                                      token_decoded == tokenizer.sep_token or token_decoded == tokenizer.cls_token or
                                      token_decoded == tokenizer.mask_token or token_decoded == tokenizer.unk_token or
                                      token_decoded == tokenizer.bos_token):
            
            token_spans.append((offset, offset))
            continue

        token_start = document[offset:].find(token_decoded)

        if token_start == -1:
            # token not found for whatever reason (should not happen if the data is clean)
            # _could_ happen in case we missed some special tokens above
            # likely cause could also be encoding errors?
            # so far, not a major problem
            current_token.append(token)
            token_decoded = tokenizer.decode(current_token)
            print(f"Token {token} ({token_decoded}) not found in text.")
            print("Trying combination of previous ones: ", current_token)
            token_start = document[offset:].find(token_decoded)

            if token_start == -1:
                token_spans.append((offset, offset))
                continue

        if token_start > 1:
            print(f"Token {token_decoded} was found skipping {token_start} characters. (text: {document[offset:][:min(20, len(document[offset:]))]})")

        if len(current_token) > 0:
            print(f"found token after combining {current_token} (len {len(current_token)}) tokens: {token_decoded}")

        token_start += offset
        token_end = token_start + len(token_decoded)
        token_spans.append((token_start, token_end))
        offset = token_end
        current_token = []

    # get ids of tokens overlapping text of span
    span_token_masks = []
    for i, span in spans.iterrows():
        span_chars = span["char_indices"]
        span_tokens = []
        for j, token_span in enumerate(token_spans):
            for char_id in span_chars:
                if char_id >= token_span[0] and char_id < token_span[1]:
                    span_tokens.append(j)
                    break
        span_token_masks.append([i in span_tokens for i in range(len(token_spans))])

    spans["token_mask"] = span_token_masks

    return spans

def torch_last_2d_max(tensor):
    """
    Get the argmax of a 3d-tensor over the last two dimensions.
    """
    flattened = tensor.reshape(tensor.shape[0], -1)
    maxima, max_ids = flattened.max(dim=-1)
    max_ids = torch.stack([max_ids // tensor.shape[1], max_ids % tensor.shape[1]], dim=-1)

    return maxima, max_ids

def get_slices(mask):
    """
    Get the slices of a boolean mask.
    """
    slices = []
    start = None
    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        if not val and start is not None:
            slices.append((start, i))
            start = None
    if start is not None:
        slices.append((start, i))
    return slices

def slice_optim_pooling(attention_matrix, relation0, relation1):
    """
    An optimized version of the avg and max calculations for the activations.
    
    Standard complex indexing creates a copy of the submatrices, which is inefficient
    and leads to memory overflow for large tensors on GPU. To avoid this, we previously
    loaded the activations to cpu and carried out the calculations there. This is slow
    and inefficent because a) we need to load the full tensor to cpu and b) the slicing
    is still done inefficiently through copying and c) the avg and max are computed
    on the cpu, too, which might potentially be slower, too.

    This function exploits the fact that most of our spans are actually continuous and
    only very rarely have gaps. Continuous slices can be computed directly on the GPU
    without copying the tensor, because torch uses a .view for slicing. Thus, for each span
    we iterate over the continuous parts and compute the avg and max pooling directly on the GPU,
    by keeping small vectors for sum, max and the position of the max value.
    """

    # relations are boolean masks, transform into slices
    relation0_slices = get_slices(relation0)
    relation1_slices = get_slices(relation1)

    # avg_vector = torch.zeros(attention_matrix.shape[0], device=attention_matrix.device)
    # max_vector = torch.zeros(attention_matrix.shape[0], device=attention_matrix.device)
    # pos_vector = torch.zeros(attention_matrix.shape[0], 2, device=attention_matrix.device)

    avg_vector = None
    max_vector = None
    pos_vector = None

    for i, slice0 in enumerate(relation0_slices):
        for j, slice1 in enumerate(relation1_slices):
            # get the submatrix
            submatrix = attention_matrix[:, slice0[0]:slice0[1]+1, slice1[0]:slice1[1]+1]

            if submatrix.storage().data_ptr() != attention_matrix.storage().data_ptr():
                print("Warning: submatrix is a copy.")
                print(submatrix.storage().data_ptr(), attention_matrix.storage().data_ptr())

            # compute the avg pooling
            avg = submatrix.sum(dim=(-1, -2))

            if avg_vector is None:
                avg_vector = avg
            else:
                avg_vector += avg

            # compute the max pooling
            max_val, max_pos = torch_last_2d_max(submatrix)
            
            if max_vector is None:
                max_vector = max_val
                pos_vector = max_pos
            else:
                is_max = max_val > max_vector
                is_max = torch.stack([is_max] * 2, dim=-1)
                # get the position of the max value
                # print(is_max.shape, pos_vector.shape, max_pos.shape)
                pos_vector = torch.where(is_max, max_pos, pos_vector)
                # print(pos_vector.shape)
                max_vector = torch.max(max_vector, max_val)

    if avg_vector is None:
        avg_vector = torch.zeros(attention_matrix.shape[0], device=attention_matrix.device)
        max_vector = torch.zeros(attention_matrix.shape[0], device=attention_matrix.device)
        pos_vector = torch.zeros(attention_matrix.shape[0], 2, device=attention_matrix.device)
    else:
        avg_vector = avg_vector / (sum(relation0) * sum(relation1))

    return avg_vector, max_vector, pos_vector
    

    

def get_max_att_span_activations(attentions, relation):
    """
    Get the max-attention-activations between two spans per attention had and layer
    and find the corresponding keys and values leading to these activations.

    Args:
        activations (list): List of torch.Tensor activations.
        span_ids (list): List of tuples with int span ids.
    """
    # the max and avg attention score for each attention head
    max_att_inter = []
    max_att_intra1 = []
    max_att_intra2 = []

    amax_att_inter = []
    amax_att_intra1 = []
    amax_att_intra2 = []

    avg_att_inter = []
    avg_att_intra1 = []
    avg_att_intra2 = []

    # relation0 = torch.tensor(relation[0]).int().to(device)
    # relation1 = torch.tensor(relation[1]).int().to(device)

    for layer_id, att in enumerate(attentions):
        # get the attention matrix for the layer
        # att = layer_att[0]

        # print(att.shape)


        t0 = time.time()
        avg_vector_inter, max_vector_inter, pos_vector_inter = slice_optim_pooling(att, relation[1], relation[0])
        avg_vector_intra1, max_vector_intra1, pos_vector_intra1 = slice_optim_pooling(att, relation[0], relation[0])
        avg_vector_intra2, max_vector_intra2, pos_vector_intra2 = slice_optim_pooling(att, relation[1], relation[1])

        # put all to cpu
        avg_vector_inter, max_vector_inter, pos_vector_inter = avg_vector_inter.cpu(), max_vector_inter.cpu(), pos_vector_inter.cpu()
        avg_vector_intra1, max_vector_intra1, pos_vector_intra1 = avg_vector_intra1.cpu(), max_vector_intra1.cpu(), pos_vector_intra1.cpu()
        avg_vector_intra2, max_vector_intra2, pos_vector_intra2 = avg_vector_intra2.cpu(), max_vector_intra2.cpu(), pos_vector_intra2.cpu()

        # print("Layer", layer_id, "took", time.time() - t0, "for optimized pooling.")

        # t0 = time.time()
        # att = att.cpu()

        # intra_att1 = att[:, relation[0]][:, :, relation[0]]
        # intra_att2 = att[:, relation[1]][:, :, relation[1]]
        

        # get lower half submatrices only since llama models have only forward attention,
        # i.e. upper half of matrix is zero
        # inter_att = att[:, relation[1]][:, :, relation[0]]

        # print("Inter att:", inter_att.shape, "Intra att1:", intra_att1.shape, "Intra att2:", intra_att2.shape)

        # print(inter_att.sum(), inter_att2.sum())
        # selected_att = torch.index_select(att, 2, relation0)
        # inter_att = torch.index_select(selected_att, 1, relation1)
        
        # print(inter_att.shape)

        # avg_att_inter.append(inter_att.reshape(inter_att.shape[0], -1).mean(dim=-1).flatten().cpu().detach())
        # avg_att_intra1.append(intra_att1.reshape(intra_att1.shape[0], -1).mean(dim=-1).flatten().cpu().detach())
        # avg_att_intra2.append(intra_att2.reshape(intra_att2.shape[0], -1).mean(dim=-1).flatten().cpu().detach())

        avg_att_inter.append(avg_vector_inter)
        avg_att_intra1.append(avg_vector_intra1)
        avg_att_intra2.append(avg_vector_intra2)

        # get max activations
        # print(inter_att)
        # max_inter, argmax_inter = torch_last_2d_max(inter_att)
        # print(max_inter, argmax_inter)
        # print("Max inter:", max_inter.shape, "Argmax inter:", argmax_inter.shape)
        # max_intra1, argmax_intra1 = torch_last_2d_max(intra_att1)
        # max_intra2, argmax_intra2 = torch_last_2d_max(intra_att2)

        # max_inter, argmax_inter = max_inter.cpu().detach(), argmax_inter.cpu().detach()
        # max_intra1, argmax_intra1 = max_intra1.cpu().detach(), argmax_intra1.cpu().detach()
        # max_intra2, argmax_intra2 = max_intra2.cpu().detach(), argmax_intra2.cpu().detach()

        # print("Max inter:", max_inter)

        # max_att_inter.append(max_inter.cpu().detach())
        # max_att_intra1.append(max_intra1.cpu().detach())
        # max_att_intra2.append(max_intra2.cpu().detach())

        max_att_inter.append(max_vector_inter)
        max_att_intra1.append(max_vector_intra1)
        max_att_intra2.append(max_vector_intra2)

        # print("Layer", layer_id, "took", time.time() - t0, "for naive pooling.")

        # print("Checking correctness:")
        # print("        avg   max   pos")
        # print("Inter:", torch.allclose(avg_vector_inter, avg_att_inter[-1]), torch.allclose(max_vector_inter, max_att_inter[-1]), torch.allclose(pos_vector_inter.int(), argmax_inter.int()))
        # print("Intra1:", torch.allclose(avg_vector_intra1, avg_att_intra1[-1]), torch.allclose(max_vector_intra1, max_att_intra1[-1]), torch.allclose(pos_vector_intra1.int(), argmax_intra1.int()))
        # print("Intra2:", torch.allclose(avg_vector_intra2, avg_att_intra2[-1]), torch.allclose(max_vector_intra2, max_att_intra2[-1]), torch.allclose(pos_vector_intra2.int(), argmax_intra2.int()))



        # first occurence of True is min index, last occurence is max index
        min_rel1 = 0
        min_rel2 = 0
        max_rel1 = 0
        max_rel2 = 0

        first = True
        for i, val in enumerate(relation[0]):
            if val and first:
                min_rel1 = i
                first = False
            if val:
                max_rel1 = i + 1
        
        first = True
        for i, val in enumerate(relation[1]):
            if val and first:
                min_rel2 = i
                max_rel2 = i
                first = False
            if val:
                max_rel2 = i + 1
    

        span_length1 = max_rel1 - min_rel1
        span_length2 = max_rel2 - min_rel2

        # print("Span lengths:", span_length1, span_length2)
        # print("Argmax inter:", argmax_inter.shape)
        # print('argmax_inter:', argmax_inter)

        # argmax_inter = torch.div(argmax_inter, torch.tensor([span_length1, span_length2]))
        if span_length1 == 0:
            span_length1 = 1
        if span_length2 == 0:
            span_length2 = 1

        argmax_inter = torch.div(pos_vector_inter, torch.tensor([span_length1, span_length2]))

        # print('after norm:', argmax_inter)

        # argmax_intra1 = torch.div(argmax_intra1, span_length1)
        # argmax_intra2 = torch.div(argmax_intra2, span_length2)
        argmax_intra1 = torch.div(pos_vector_intra1, span_length1)
        argmax_intra2 = torch.div(pos_vector_intra2, span_length2)

        amax_att_inter.append(argmax_inter.cpu().detach())
        amax_att_intra1.append(argmax_intra1.cpu().detach())
        amax_att_intra2.append(argmax_intra2.cpu().detach())


    max_att_inter = torch.cat(max_att_inter, dim=0)
    max_att_intra1 = torch.cat(max_att_intra1, dim=0)
    max_att_intra2 = torch.cat(max_att_intra2, dim=0)
    
    amax_att_inter = torch.cat(amax_att_inter, dim=0)
    amax_att_intra1 = torch.cat(amax_att_intra1, dim=0)
    amax_att_intra2 = torch.cat(amax_att_intra2, dim=0)

    avg_att_inter = torch.cat(avg_att_inter, dim=0)
    avg_att_intra1 = torch.cat(avg_att_intra1, dim=0)
    avg_att_intra2 = torch.cat(avg_att_intra2, dim=0)

    # print("max_inter", max_att_inter.shape, "max_intra1", max_att_intra1.shape, "max_intra2", max_att_intra2.shape)
    # print("amax_inter", amax_att_inter.shape, "amax_intra1", amax_att_intra1.shape, "amax_intra2", amax_att_intra2.shape)
    # print("avg_inter", avg_att_inter.shape, "avg_intra1", avg_att_intra1.shape, "avg_intra2", avg_att_intra2.shape)


    max_dict = {"inter": max_att_inter, "intra1": max_att_intra1, "intra2": max_att_intra2}
    avg_dict = {"inter": avg_att_inter, "intra1": avg_att_intra1, "intra2": avg_att_intra2}
    pos_dict = {"inter": amax_att_inter, "intra1": amax_att_intra1, "intra2": amax_att_intra2}

    return max_dict, avg_dict, pos_dict


def load_data(path, partition):
    docs = pd.read_csv(path + f"documents_{partition}.csv")
    spans = pd.read_csv(path + f"spans_{partition}.csv")
    relations = pd.read_csv(path + f"relations_{partition}.csv")

    # convert char_indices to list of ints
    spans["char_indices"] = spans["char_indices"].map(lambda x: [int(i) for i in re.findall(r"\d+", x)])

    spans["intra_attention_embedding_avg"] = None
    spans["intra_attention_embedding_max"] = None
    spans["intra_attention_embedding_pos"] = None

    relations["inter_attention_embedding_avg"] = None
    relations["inter_attention_embedding_max"] = None
    relations["inter_attention_embedding_pos"] = None

    return docs, spans, relations

def test():
    # for debugging purposes
    document = "We didn't do a whole lot of hiking here because there was not a whole " + \
                 "lot of cloud coverage that day, and it was so hot."
    spans = pd.DataFrame({"char_indices": [list(range(0, 39)) , list(range(39, len(document)))]})

def add_col_vals_and_store_pt_vector(col, df, vals, add_vals):

    # make sure to keep old values if they exist
    if col in df.columns:
        df["temp"] = df[col]
    else:
        df["temp"] = None


    add = lambda x: add_vals(x, vals)

    df[col] = df.apply(add, axis=1)
    # torch.tensor(spans[col].tolist()).to(path + f"{col}_checkpoint.pt")

    return df

def store_checkpoint(spans, relations, path, intra, inter):
    print("Storing spans...")

    def add_span(x, vals):

        if x["doc_id"] in vals.keys() and x["unit_id"] in vals[x["doc_id"]].keys():
            v = vals[x["doc_id"]][x["unit_id"]]
            if v.dtype == torch.bfloat16:
                v = v.float()
            return v
        return x["temp"]
    
    spans = add_col_vals_and_store_pt_vector("intra_attention_embedding_avg", spans, intra["avg_pooling"], add_span)
    spans = add_col_vals_and_store_pt_vector("intra_attention_embedding_max", spans, intra["max_pooling"], add_span)
    spans = add_col_vals_and_store_pt_vector("intra_attention_embedding_pos", spans, intra["pos_pooling"], add_span)

    print("Storing rels...")

    def add_rel(x, vals):
        if x["doc_id"] in vals.keys() and (x["unit1"], x["unit2"]) in vals[x["doc_id"]].keys():
            v = vals[x["doc_id"]][(x["unit1"], x["unit2"])]
            if v.dtype == torch.bfloat16:
                v = v.float()
            return v
        return x["temp"]

    relations = add_col_vals_and_store_pt_vector("inter_attention_embedding_avg", relations, inter["avg_pooling"], add_rel)
    relations = add_col_vals_and_store_pt_vector("inter_attention_embedding_max", relations, inter["max_pooling"], add_rel)
    relations = add_col_vals_and_store_pt_vector("inter_attention_embedding_pos", relations, inter["pos_pooling"], add_rel)

    spans["computed"] = spans["intra_attention_embedding_avg"].isnull() == False

    print("Spans computed:", sum(spans["computed"]))

    print(spans.head())

    computed = spans[spans["computed"]]
    print(computed.head())
    print(computed.shape)

    spans_avg = torch.stack(computed["intra_attention_embedding_avg"].values.tolist()).cpu()
    spans_max = torch.stack(computed["intra_attention_embedding_max"].values.tolist()).cpu()
    spans_pos = torch.stack(computed["intra_attention_embedding_pos"].values.tolist()).cpu()

    # make a copy and store
    spans_copy = spans.copy()
    spans_copy["temp"] = None
    spans_copy["intra_attention_embedding_avg"] = None
    spans_copy["intra_attention_embedding_max"] = None
    spans_copy["intra_attention_embedding_pos"] = None

    print(spans_copy.columns)
    for col in spans_copy.columns:
        print(col, spans_copy[col].head())

    torch.save(spans_avg, path + "intra_attention_embedding_avg_checkpoint.pt")
    torch.save(spans_max, path + "intra_attention_embedding_max_checkpoint.pt")
    torch.save(spans_pos, path + "intra_attention_embedding_pos_checkpoint.pt")
    
    spans_copy.to_pickle(path + "spans_encoded_checkpoint.pkl")

    relations["computed"] = relations["inter_attention_embedding_avg"].isnull() == False

    print("Relations computed:", sum(relations["computed"]))

    computed = relations[relations["computed"]]

    print(relations.head())

    rels_avg = torch.stack(computed["inter_attention_embedding_avg"].values.tolist()).cpu()
    rels_max = torch.stack(computed["inter_attention_embedding_max"].values.tolist()).cpu()
    rels_pos = torch.stack(computed["inter_attention_embedding_pos"].values.tolist()).cpu()

    # print(rels_avg.shape)
    # print(rels_max.shape)
    # print(rels_pos.shape)

    relations_copy = relations.copy()
    relations_copy["temp"] = None
    relations_copy["inter_attention_embedding_avg"] = None
    relations_copy["inter_attention_embedding_max"] = None
    relations_copy["inter_attention_embedding_pos"] = None

    torch.save(rels_avg, path + "inter_attention_embedding_avg_checkpoint.pt")
    torch.save(rels_max, path + "inter_attention_embedding_max_checkpoint.pt")
    torch.save(rels_pos, path + "inter_attention_embedding_pos_checkpoint.pt")

    relations_copy.to_pickle(path + "relations_encoded_checkpoint.pkl")

    return spans, relations, {"avg_pooling": {}, "max_pooling": {}, "pos_pooling": {}}, {"avg_pooling": {}, "max_pooling": {}, "pos_pooling": {}}

def get_windows(all_doc_spans, all_doc_relations, tokens, max_length):
    """
    Split the document into windows of max_length with a stride of max_length // 2.
    Map relations to the windows where they are closest to the middle.
    """
    windows = []
    stride = max_length // 2
    last_window = False


    if len(tokens["input_ids"][0]) <= max_length:
        return [(all_doc_spans, all_doc_relations, tokens)]
    
    print(f"Splitting document into {len(tokens['input_ids'][0]) / stride} windows...")

    first_window = True

    for i in range(0, len(tokens["input_ids"][0]), stride):

        window_start = i
        window_end = window_start + max_length

        if window_end > len(tokens["input_ids"][0]):
            window_end = len(tokens["input_ids"][0])
            window_start = window_end - max_length

        # only keep spans present in window
        # print("Window:", window_start, window_end)
        
        #window_spans = doc_spans[(doc_spans["char_indices"].apply(lambda x: x[0]) >= window_start) &
        #                            (doc_spans["char_indices"].apply(lambda x: x[1]) <= window_end)]
        window_spans = all_doc_spans.copy()
        window_spans["span_length"] = window_spans["token_mask"].apply(lambda x: sum(x))
        window_spans["token_mask"] = window_spans["token_mask"].apply(lambda mask: mask[window_start:window_end])
        
        window_spans = window_spans[window_spans["token_mask"].apply(lambda x: sum(x)) == window_spans["span_length"]]
        

        # window_spans["char_indices"] = window_spans["char_indices"].apply(lambda x: [x - window_start for x in x])

        doc_relations = all_doc_relations.copy()

        window_relations = doc_relations[(doc_relations["unit1"].isin(window_spans["unit_id"])) &
                                            (doc_relations["unit2"].isin(window_spans["unit_id"]))]
        
        # window_relations["center"] = window_relations.apply(lambda x: (max(window_spans[window_spans["unit_id"] == x["unit1"]]["char_indices"].values[0]) + \
        #                                                                 min(window_spans[window_spans["unit_id"] == x["unit2"]]["char_indices"].values[0])) / 2, axis=1)
        
        # remove relations whose center is closer to the end than to the middle of the window (i.e. should be in the next window)
        # if not last_window:
        #     window_relations = window_relations[window_relations["char_indices"] < window_end - (max_length / 4)]
            # remove spans not involved in a relation
        #     window_spans = window_spans[window_spans["unit_id"].isin(window_relations["unit1"]) | window_spans["unit_id"].isin(window_relations["unit2"])]
            
        # if not first_window:
        #     window_relations = window_relations[window_relations["center"] > window_start + (max_length / 4)]
        #     window_spans = window_spans[window_spans["unit_id"].isin(window_relations["unit1"]) | window_spans["unit_id"].isin(window_relations["unit2"])]

        window_tokens = {"input_ids": tokens["input_ids"][:, window_start:window_end],
                        "attention_mask": tokens["attention_mask"][:, window_start:window_end],
                        }
        first_window = False
        
        windows.append((window_spans, window_relations, window_tokens))

    return windows[::-1]

def encode_all_docs(docs, spans, relations, tokenizer, model, checkpoint_path=None, max_length=4000, min_length=0, emb_dim=1024, stop_after=100000):

    print("Encoding documents...")

    counter = -1
    preds = 0

    new_checkpoint = False

    intra = {"max_pooling": {}, "avg_pooling": {}, "pos_pooling": {}}
    inter = {"max_pooling": {}, "avg_pooling": {}, "pos_pooling": {}}
    
    for i, doc in docs.iterrows():

        counter += 1

        document = doc["document"]
        doc_spans = spans[spans["doc_id"] == doc["doc_id"]]

        if "intra_attention_embedding_avg" in spans.columns and "inter_attention_embedding_avg" in relations.columns:
            if doc_spans["intra_attention_embedding_avg"].isnull().sum() < len(doc_spans):
                if doc_spans["intra_attention_embedding_max"].isnull().sum() > 0:
                    print(f"Document {doc['doc_id']} has some spans already encoded, but not all.")
                    print(doc_spans["intra_attention_embedding_max"].isnull().sum(), "missing spans.")
                    print("Missing spans:", doc_spans[doc_spans["intra_attention_embedding_avg"].isnull()].head())
                    # missing vals are likely due to relations not captured by windowing approach
                print(f"Document already encoded, skipping... ({counter})", end="\r")
                continue

        tokens = tokenizer(document, return_tensors="pt").to(device)

        if len(tokens["input_ids"][0]) > max_length:
            print()
            print(f"Document #{counter} longer than {max_length} tokens ({len(tokens["input_ids"][0])}), skipping...")
            print("Doc ID:", doc["doc_id"])
            # continue

        if len(tokens["input_ids"][0]) < min_length:
            print(f"Document #{counter} shorter than {min_length} tokens ({len(tokens["input_ids"][0])}), skipping...")
            print("Doc ID:", doc["doc_id"])
            continue
        

        doc_spans = add_span_masks(tokens["input_ids"][0], document, doc_spans, tokenizer)
        doc_relations = relations[relations["doc_id"] == doc["doc_id"]]
        # TODO: Get the following into an embed_document function
        # split doc into windows of max_length with a stride of max_length // 2
        # map relations to the windows where they are closest to the middle
        # encode the windows and store the embeddings in the spans dataframe as before

        windows = get_windows(doc_spans, doc_relations, tokens, max_length)
        if len(windows) > 1: 
            print("Attempting to encode very long document in parts...")
            print("Doc ID:", doc["doc_id"])
            print(len(windows))

        ds, dr, toks = windows[0]
        # print("First window:", len(toks["input_ids"]))
        encoded = None
        encoded = encode_document(ds, dr, toks, model)
        i = 1
        while i < len(windows):
            ds, dr, toks = windows[i]
            encoded = encode_document(ds, dr, toks, model, **encoded)
            i += 1

        intra["max_pooling"][doc["doc_id"]] = encoded["intra_max_attentions"]
        intra["avg_pooling"][doc["doc_id"]] = encoded["intra_avg_attentions"]
        intra["pos_pooling"][doc["doc_id"]] = encoded["intra_amax_attentions"] 
        inter["max_pooling"][doc["doc_id"]] = encoded["inter_max_attentions"]
        inter["avg_pooling"][doc["doc_id"]] = encoded["inter_avg_attentions"]
        inter["pos_pooling"][doc["doc_id"]] = encoded["inter_amax_attentions"]

        print("Current doc:", counter, end="\r")


        if preds > stop_after:
            return spans, relations


        if (counter % 100 == 0 or i % 1000 == 0) and checkpoint_path is not None and counter > 0:
            # print(relations.head())
            print(f"Saving progress... ({counter})")
            spans, relations, intra, inter = store_checkpoint(spans, relations, checkpoint_path, intra, inter)
            print("After checkpointing")
            print(spans.head())
            print(relations.head())
            new_checkpoint = False

        if counter % 50 == 0:
            print("Current doc:", counter)


    spans, relations, intra, inter = store_checkpoint(spans, relations, checkpoint_path, intra, inter)

    return spans, relations

def encode_document(
    doc_spans, doc_relations, tokens, model,
    intra_max_attentions={},
    inter_max_attentions={},
    intra_avg_attentions={},
    inter_avg_attentions={},
    intra_amax_attentions={},
    inter_amax_attentions={},
):
    last_states, output = None, None
    last_states, output, tokens = get_outputs(model, tokens)


    # zero out diagonals
    for layer_id, layer_att in enumerate(output):
        # get the attention matrix for the layer
        att = layer_att[0]

        # fill diagonals with zeros to ignore self-attention
        att = att.view(-1, att.size(-1))
        att.fill_diagonal_(0)
        att = att.view(layer_att[0].shape)

        output[layer_id] = att

    for i, relation in doc_relations.iterrows():
        r1 = doc_spans[relation["unit1"] == doc_spans["unit_id"]]["token_mask"].values[0]
        r2 = doc_spans[relation["unit2"] == doc_spans["unit_id"]]["token_mask"].values[0]
        # print("Relation:", relation["unit1"], relation["unit2"])

        relation_masks = (r1, r2)

        # exchange this function for other strategies than "max-pooling"
        scores, avg, pos = get_max_att_span_activations(output, relation_masks)

        # keys and vals actually are too high dimensional
        intra_max_attentions[relation["unit1"]] = scores["intra1"]  # , keys["intra1"], vals["intra1"])
        intra_max_attentions[relation["unit2"]] = scores["intra2"]  #, keys["intra2"], vals["intra2"])
        inter_max_attentions[(relation["unit1"], relation["unit2"])] = scores["inter"]     # keys["inter"], vals["inter"])

        intra_avg_attentions[relation["unit1"]] = avg["intra1"]
        intra_avg_attentions[relation["unit2"]] = avg["intra2"]
        inter_avg_attentions[(relation["unit1"], relation["unit2"])] = avg["inter"]

        intra_amax_attentions[relation["unit1"]] = pos["intra1"]
        intra_amax_attentions[relation["unit2"]] = pos["intra2"]
        inter_amax_attentions[(relation["unit1"], relation["unit2"])] = pos["inter"]
    
    return {
        "intra_max_attentions": intra_max_attentions,
        "inter_max_attentions": inter_max_attentions,
        "intra_avg_attentions": intra_avg_attentions,
        "inter_avg_attentions": inter_avg_attentions,
        "intra_amax_attentions": intra_amax_attentions,
        "inter_amax_attentions": inter_amax_attentions
    }

def load_checkpoint(path, spans, relations):
    # check if a checkpoint exists, if so, load it
    if os.path.exists(path + "temp/spans_encoded_checkpoint.pkl"):
        print("Loading checkpoint...")
        spans = pd.read_pickle(path + "temp/spans_encoded_checkpoint.pkl")
        relations = pd.read_pickle(path + "temp/relations_encoded_checkpoint.pkl")

        intra_avg = torch.load(path + "temp/intra_attention_embedding_avg_checkpoint.pt")
        intra_max = torch.load(path + "temp/intra_attention_embedding_max_checkpoint.pt")
        intra_pos = torch.load(path + "temp/intra_attention_embedding_pos_checkpoint.pt")

        inter_avg = torch.load(path + "temp/inter_attention_embedding_avg_checkpoint.pt")
        inter_max = torch.load(path + "temp/inter_attention_embedding_max_checkpoint.pt")
        inter_pos = torch.load(path + "temp/inter_attention_embedding_pos_checkpoint.pt")

        intra_avg_generator = (v for v in intra_avg)
        spans["intra_attention_embedding_avg"] = spans["computed"].apply(lambda x: next(intra_avg_generator) if x else None)
        intra_max_generator = (v for v in intra_max)
        spans["intra_attention_embedding_max"] = spans["computed"].apply(lambda x: next(intra_max_generator) if x else None)
        intra_pos_generator = (v for v in intra_pos)
        spans["intra_attention_embedding_pos"] = spans["computed"].apply(lambda x: next(intra_pos_generator) if x else None)
        spans["temp"] = None


        inter_avg_generator = (v for v in inter_avg)
        relations["inter_attention_embedding_avg"] = relations["computed"].apply(lambda x: next(inter_avg_generator) if x else None)
        inter_max_generator = (v for v in inter_max)
        relations["inter_attention_embedding_max"] = relations["computed"].apply(lambda x: next(inter_max_generator) if x else None)
        inter_pos_generator = (v for v in inter_pos)
        relations["inter_attention_embedding_pos"] = relations["computed"].apply(lambda x: next(inter_pos_generator) if x else None)
        relations["temp"] = None


    else:
        print("No checkpoint found, starting from scratch...")
    
    return spans, relations

if __name__ == "__main__":    

    path = "data/disrpt_private/rel_embeddings_llama3_8B/"
    model_name = "meta-llama/Meta-Llama-3-8B"

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

    #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    spans, relations = encode_all_docs(docs, spans, relations, tokenizer, model, checkpoint_path=path+"temp/",
                                            min_length=0, max_length=3800)

    # print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30))
    # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))

    spans.to_pickle(path + "spans_train_encoded.pkl")
    relations.to_pickle(path + "relations_train_encoded.pkl")

    spans_test, relations_test = encode_all_docs(docs_test, spans_test, relations_test, tokenizer, model, 
                                                 checkpoint_path=path+"temp/test/", max_length=3800)

    spans_test.to_pickle(path + "spans_test_encoded.pkl")
    relations_test.to_pickle(path + "relations_test_encoded.pkl")

import pandas as pd
import os
import io
import re
import numpy as np


class DISRPTReader:
    def __init__(self, path):
        self.path = path
        self.path_conllu = path
        self.corpora_paths = os.listdir(path)
        self.corpora = {}
        self.data = {"train": {}, "dev": {}, "test": {}}
        self.HEADER = "doc\tunit1_toks\tunit2_toks\tunit1_txt\tunit2_txt\ts1_toks\ts2_toks\tunit1_sent\tunit2_sent\tdir\torig_label\tlabel" # "doc\tunit1_toks\tunit2_toks\tunit1_txt\tunit2_txt\tu1_raw\tu2_raw\ts1_toks\ts2_toks\tunit1_sent\tunit2_sent\tdir\trel_type\torig_label\tlabel"
        self.LABEL_ID = -1
        self.TYPE_ID = -3
        self.DISRPT_TYPES = ['Implicit', 'Explicit', 'AltLex', 'AltLexC', 'Hypophora']
        self.TEXT1_ID = 5
        self.TEXT2_ID = 6
        self.ignore = [] # ["data/disrpt_private/data/eng.rst.gum"]
        self.docs = {}
        self.documents = None
        self.spans = None
        self.relations = None
        self.rels_data = None
        

    def read(self, partitions=["train", "dev"]):
        for corpus_path in self.corpora_paths:
            full_path = os.path.join(self.path, corpus_path)
            if full_path in self.ignore:
                print(f"Skipping {full_path}")
                continue
            if not os.path.isdir(full_path):
                print(f"{full_path} is not a directory, skipping...")
                continue
            print(f"Reading corpus {full_path}")
            self.corpora[full_path] = os.listdir(full_path)
            for partition in partitions:
                self._read_partition(corpus_path, partition)

        self.rels_data = self._get_rels_dataframe()
        self.documents, self.spans, self.relations = self._get_complete_samples()
        

    def _read_partition(self, corpus_path, partition):
        rels_partition_path = os.path.join(self.path, corpus_path, corpus_path + "_" + partition + ".rels")
        print(f"Reading partition {rels_partition_path} ({partition})")
        
        if not os.path.exists(rels_partition_path):
            print(f"Partition {rels_partition_path} does not exist, skipping...")
        else:
            # load rels file
            rels = self._parse_rels_data(rels_partition_path, str_i=False, rel_t=True)
            rels["corpus"] = corpus_path
            rels["partition"] = partition
            self.data[partition][corpus_path] = rels
        
        # conllu_partition_path = os.path.join(self.path, corpus_path, corpus_path + "_" + partition + ".conllu")

        # if not os.path.exists(conllu_partition_path):
        #     print(f"Partition {conllu_partition_path} does not exist, skipping...")
        # else:
        #     # load conllu file
        #     self.docs = self._parse_conllu_data(conllu_partition_path, str_i=False)


    def _get_rels_dataframe(self):
        # combine single datasets into one dataframe
        dataframes = []
        for partition in self.data:
            for corpus in self.data[partition]:
                dataframes.append(self.data[partition][corpus])

        df = pd.concat(dataframes, ignore_index=True)

        print(df.head())
        print(df.shape)

        # for single token units, change to tok-tok
        df["unit1"] = df["unit1"].apply(lambda x: x if "-" in x else x + "-" + x)
        df["unit2"] = df["unit2"].apply(lambda x: x if "-" in x else x + "-" + x)


        df = df[(df["unit1"] != "_-_") & (df["unit2"] != "_-_")]
        
        # add int token unit ids
        df["unit1_start"] = df["unit1"].apply(lambda x: int(re.sub(r',[0-9]+', "", re.sub(r'-[0-9]+,[0-9]+-', "-", x)).replace(",", "-").strip().split("-")[0]))
        df["unit1_end"] = df["unit1"].apply(lambda x: int(re.sub(r',[0-9]+', "", re.sub(r'-[0-9]+,[0-9]+-', "-", x)).replace(",", "-").strip().split("-")[-1]))
        
        df["unit1_gap"] = df["unit1"].apply(lambda x: re.findall(r'([0-9]+),([0-9]+)', x)[0] if re.findall(r'([0-9]+,[0-9]+)', x) else None)
        df["unit1_gap_start"] = df["unit1_gap"].apply(lambda x: int(x[0]) if x is not None else None)
        df["unit1_gap_end"] = df["unit1_gap"].apply(lambda x: int(x[1]) if x is not None else None)
                                            

        df["unit2_start"] = df["unit2"].apply(lambda x: int(re.sub(r',[0-9]+', "", re.sub(r'-[0-9]+,[0-9]+-', "-", x)).strip().split("-")[0]))
        df["unit2_end"] = df["unit2"].apply(lambda x: int(re.sub(r',[0-9]+', "", re.sub(r'-[0-9]+,[0-9]+-', "-", x)).replace(",", "-").strip().split("-")[-1]))
        df["unit2_gap"] = df["unit2"].apply(lambda x: re.findall(r'([0-9]+),([0-9]+)', x)[0] if re.findall(r'([0-9]+,[0-9]+)', x) else None)
        df["unit2_gap_start"] = df["unit2_gap"].apply(lambda x: int(x[0]) if x is not None else None)
        df["unit2_gap_end"] = df["unit2_gap"].apply(lambda x: int(x[1]) if x is not None else None)

        df["sent1_start"] = df["sent1_toks"].apply(lambda x: int(x.replace(",", "-").split("-")[0]))
        df["sent1_end"] = df["sent1_toks"].apply(lambda x: int(x.replace(",", "-").split("-")[-1]))

        df["sent2_start"] = df["sent2_toks"].apply(lambda x: int(x.replace(",", "-").split("-")[0]))
        df["sent2_end"] = df["sent2_toks"].apply(lambda x: int(x.replace(",", "-").split("-")[-1]))

        return df
    
    def remove_zh_whitespace(self, text):
        # remove whitespace only between characters that are both not in [a-zA-Z] and do not come
        # in between a punctuation mark and [a-zA-Z]
        text = re.sub(r'\s(?=[^a-zA-Z])|(?<=[^a-zA-Z\.\:\;\,\!\?\)\"\'\]\}\>])\s(?=[a-zA-Z])', "", text)
        # text = re.sub(r'(?<=[\.\:\;\,\!\?\)\"\'\]\}\>])\s(?=[^a-zA-Z])', "", text)
        return text
        
    def _parse_rels_data(self, path, str_i, rel_t):
        """
		Rels format from DISRPT = header, then one relation classification instance per line. 
		:LREC_2024_header = 15 columns.
		"""
        data = self._get_data(path, str_i)
        header = data.split("\n")[0]

        if header != self.HEADER:
            print(f"Unrecognized .rels header in {path}.")
            print(header)
            print(self.HEADER)

        # assert header == self.HEADER, "Unrecognized .rels header."

        #column_ID = self.TYPE_ID if rel_t == True else self.LABEL_ID

        # if re.findall(r".gum", path) and header != self.HEADER:
        # need different parsing for GUM
        # gum_header = "doc\tunit1_toks\tunit2_toks\tunit1_txt\tunit2_txt\ts1_toks\ts2_toks\tunit1_sent\tunit2_sent\tdir\torig_label\tlabel"

        # if header == gum_header:
        #    print("GUM header detected.")

        rels = data.split("\n")[1:]
        labels = [line.split("\t")[-1] for line in rels]
        types = [line.split("\t")[-2] for line in rels] if rel_t else []
        identifier = [str(path)+"/"+line.split("\t")[0] for line in rels]
        unit1 = [line.split("\t")[1] for line in rels]
        unit2 = [line.split("\t")[2] for line in rels]
        unit1_sent = [line.split("\t")[7] for line in rels]
        unit2_sent = [line.split("\t")[8] for line in rels]
        text1 = [line.split("\t")[3] for line in rels]
        text2 = [line.split("\t")[4] for line in rels]
        sent1_toks = [line.split("\t")[5] for line in rels]
        sent2_toks = [line.split("\t")[6] for line in rels]


        # delete whitespace in Chinese samples
        if "zho." in path:
            text1 = [self.remove_zh_whitespace(text) for text in text1]
            text2 = [self.remove_zh_whitespace(text) for text in text2]
            unit1_sent = [self.remove_zh_whitespace(text) for text in unit1_sent]
            unit2_sent = [self.remove_zh_whitespace(text) for text in unit2_sent]


        """  
        else:

            rels = data.split("\n")[1:]
            labels = [line.split("\t")[self.LABEL_ID] for line in rels] ######## .lower()
            types = [line.split("\t")[self.TYPE_ID] for line in rels] if rel_t else []
            identifier = [str(path)+"/"+line.split("\t")[0] for line in rels]
            unit1 = [line.split("\t")[1] for line in rels]
            unit2 = [line.split("\t")[2] for line in rels]
            unit1_sent = [line.split("\t")[9] for line in rels]
            unit2_sent = [line.split("\t")[10] for line in rels]

            text1 = [line.split("\t")[self.TEXT1_ID] for line in rels]
            text2 = [line.split("\t")[self.TEXT2_ID] for line in rels]

            sent1_toks = [line.split("\t")[7] for line in rels]
            sent2_toks = [line.split("\t")[8] for line in rels]
        """

        df = pd.DataFrame({"identifier": identifier, "unit1": unit1, "unit2": unit2, "label": labels,
                            "type": types, "text1": text1, "text2": text2, 
                            "sent1": unit1_sent, "sent2": unit2_sent, "sent1_toks": sent1_toks, "sent2_toks": sent2_toks})

        return df
    
    def _parse_conllu_data(self, path:str, str_i:bool) -> tuple[list, list, list]:
        """
        LABEL = in last column
        """
        
        data = self._get_data(path, str_i)
        docs = self.docs

        current_doc_id = None

        for line in data.split("\n"):  # this loop is same than version 1
            if line.startswith("#"):
                if "doc_id" in line or "doc" in line and current_doc_id is None:
                    current_doc_id = line.split(" = ")[1].strip()
                    if current_doc_id not in docs.keys():
                        docs[current_doc_id] = []
                if "sent_id" in line:
                    current_doc_id = re.findall(r"(.+)\-[0-9]+", line.split(" = ")[1])[0]
                    if current_doc_id not in docs.keys():
                        docs[current_doc_id] = []
                continue

            if line == "":
                continue

            fields = line.split("\t") # Token

            
            if "-" in fields[0] or "." in fields[0]:  # Multi-Word Expression or Ellips : No pred shall be there....
                continue
            
            if current_doc_id is None:
                print("Error: current_doc_id is None, skipping...")
                print(line)
                continue
            docs[current_doc_id].append(fields[1])


        return docs

    
    def _get_data(self, infile: str, str_i=False) -> str:
        """
        Stock data from file or stream.
        """
        if not str_i:
            data = io.open(infile, encoding="utf-8").read().strip().replace("\r", "")
        else:
            data = infile.strip()
        return data
    
    def _get_complete_samples(self):
        """
        Build complete samples from the rels data.
        """

        documents = []
        all_relations = []
        all_spans = []

        for doc_id in self.rels_data["identifier"].unique():
            relevant_rels = self.rels_data[self.rels_data["identifier"] == doc_id]
            
            if relevant_rels.shape[0] == 0:
                print(f"Doc {doc_id} has no relevant relations, skipping...")
                continue

            spans = []
            relations = []
            sentences = {}

            for i, rel in relevant_rels.iterrows():
                spans.append({"doc_id": doc_id, "unit_id": rel["unit1"], "text": rel["text1"], "unit_start": rel["unit1_start"], 
                              "unit_end": rel["unit1_end"], "unit_gap_start": rel["unit1_gap_start"], "unit_gap_end": rel["unit1_gap_end"]})
                spans.append({"doc_id": doc_id, "unit_id": rel["unit2"], "text": rel["text2"], "unit_start": rel["unit2_start"],
                              "unit_end": rel["unit2_end"], "unit_gap_start": rel["unit2_gap_start"], "unit_gap_end": rel["unit2_gap_end"]})
                relations.append({"doc_id": doc_id, "unit1": rel["unit1"], "unit2": rel["unit2"], "label": rel["label"], "type": rel["type"]})
                sentences[rel["sent1_start"]] = rel["sent1"]
                sentences[rel["sent2_start"]] = rel["sent2"]

            spans = pd.DataFrame(spans)
            spans = spans.drop_duplicates()

            relations = pd.DataFrame(relations)

            # build document from sentences
            jstring = "" if "zho." in doc_id else " "
            document = jstring.join([sentences[key] for key in sorted(sentences.keys())])

            char_indices = {}

            for i, span in spans.iterrows():
                # match every span to its character indices in document
                chars = []
                # spans can have gaps indicated by <*> in the text
                for subspan in span["text"].split("<*>"):
                    start = document.find(subspan)
                    end = start + len(subspan)
                    chars += list(range(start, end))
                char_indices[span["unit_id"]] = chars

            spans["char_indices"] = spans["unit_id"].apply(lambda x: char_indices[x])

            all_spans.append(spans)
            all_relations.append(relations)
            documents.append({"doc_id": doc_id, "document": document})

        spans = pd.concat(all_spans, ignore_index=True)
        relations = pd.concat(all_relations, ignore_index=True)
        documents = pd.DataFrame(documents)

        return documents, spans, relations


        
    

if __name__ == "__main__":

    print("Reading train+dev set...")
    reader = DISRPTReader("data/disrpt_private/data/")
    reader.read()
    print(reader.documents.head())
    print(reader.spans.head())
    print(reader.relations.head())

    # store to rel_embeddings
    reader.documents.to_csv("data/disrpt_private/rel_embeddings/documents_train.csv")
    reader.spans.to_csv("data/disrpt_private/rel_embeddings/spans_train.csv")
    reader.relations.to_csv("data/disrpt_private/rel_embeddings/relations_train.csv")

    print("Reading test set...")

    reader_test = DISRPTReader("data/disrpt_private/data/")
    reader_test.read(partitions=["test"])

    print(reader.documents.head())
    print(reader.spans.head())
    print(reader.relations.head())

    reader_test.documents.to_csv("data/disrpt_private/rel_embeddings/documents_test.csv")
    reader_test.spans.to_csv("data/disrpt_private/rel_embeddings/spans_test.csv")
    reader_test.relations.to_csv("data/disrpt_private/rel_embeddings/relations_test.csv")


    



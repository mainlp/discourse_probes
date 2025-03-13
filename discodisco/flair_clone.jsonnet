local transformer_model_name = std.extVar("EMBEDDING_MODEL_NAME");
local corpus_name = std.extVar("CORPUS");
local embedding_dim = std.parseInt(std.extVar("EMBEDDING_DIMS"));  # uniquely determined by transformer_model

local features = {
  "features": {
    "nuc_children": {"source_key": "nuc_children"},
    "sat_children": {"source_key": "sat_children"},
    "genre": {"source_key": "genre", "label_namespace": "genre"},
    "u1_discontinuous": {"source_key": "u1_discontinuous", "label_namespace": "discontinuous"},
    "u2_discontinuous": {"source_key": "u2_discontinuous", "label_namespace": "discontinuous"},
    "u1_issent": {"source_key": "u1_issent", "label_namespace": "issent"},
    "u2_issent": {"source_key": "u2_issent", "label_namespace": "issent"},
    "unit1_case": {"source_key": "unit1_case", "label_namespace": "case"},
    "unit2_case": {"source_key": "unit2_case", "label_namespace": "case"},
    "u1_depdir": {"source_key": "u1_depdir", "label_namespace": "depdir"},
    "u2_depdir": {"source_key": "u2_depdir", "label_namespace": "depdir"},
    "u1_func": {"source_key": "u1_func", "label_namespace": "func"},
    "u2_func": {"source_key": "u2_func", "label_namespace": "func"},
    "length_ratio": {"source_key": "length_ratio"},
    //"same_speaker": {"source_key": "same_speaker", "label_namespace": "same_speaker"},
    "doclen": {"source_key": "doclen"},
    //"distance": {"source_key": "distance"},
    "distance": {
      "source_key": "distance",
      "xform_fn": {
        "type": "bins",
        "bins": [[-1e9, -8], [-8, -2], [-2, 0], [0, 2], [2, 8], [8, 1e9]]
      },
      "label_namespace": "distance"
    },
    "u1_position": {
    "source_key": "u1_position",
    "xform_fn": {
      "type": "bins",
      "bins": [[0.0, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6], [0.6, 0.7], [0.7, 0.8], [0.8, 0.9], [0.9, 1.0], [1.0, 1e9]]
      //"bins": [[0.0, 0.2], [0.2, 0.4], [0.4, 0.6], [0.6, 0.8], [0.8, 1.0], [1.0, 1e9]]
    },
    "label_namespace": "u1_position"
    }, 
    "u2_position": {
    "source_key": "u2_position",
    "xform_fn": {
      "type": "bins",
      "bins": [[0.0, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6], [0.6, 0.7], [0.7, 0.8], [0.8, 0.9], [0.9, 1.0], [1.0, 1e9]]
      //"bins": "bins": [[0.0, 0.2], [0.2, 0.4], [0.4, 0.6], [0.6, 0.8], [0.8, 1.0], [1.0, 1e9]]
    },
    "label_namespace": "u2_position"
    },
    "lex_overlap_length": {
      "source_key": "lex_overlap_length",
      "xform_fn": {
        "type": "bins",
        "bins": [[0, 2], [2, 7], [7, 1e9]]
      },
      "label_namespace": "lex_overlap"
    }
  },
  "corpus": corpus_name,
  // By default, we will use all features for a corpus, but they can be overridden below.
  // The values inside the array need to match a key under the "features" dict above.
  "corpus_configs": {
    "mul.dis.all": [], 
    "deu.rst.pcc": [],
    "eng.pdtb.pdtb": [], 
    "eng.rst.gum": [], 
    "eng.rst.rstdt": [],
    "eng.sdrt.stac": [],
    "eus.rst.ert": [],
    "fra.sdrt.annodis": [],
    "nld.rst.nldt": [],
    "por.rst.cstn": [],
    "rus.rst.rrt": [],
    "spa.rst.rststb": [],
    "spa.rst.sctb": [], 
    "tur.pdtb.tdb": [], 
    "zho.pdtb.cdtb": [],
    "zho.rst.sctb": [],
    "fas.rst.prstc": [], 
    // newly added corpora
    "eng.dep.covdtb": [],
    "eng.dep.scidtb": [], 
    "eng.pdtb.tedm": [], 
    "tur.pdtb.tedm": [],
    "zho.rst.gcdt": [],
    "ita.pdtb.luna": [], 
    "por.pdtb.crpc": [],
    "por.pdtb.tedm": [],
    "tha.pdtb.tdtb": [],
    "zho.dep.scidtb": [],
  }
};

// For small corpora, make this number reflect the size of train
// For larger corpora, use a smaller number, aiming for 1/3 of total size
local batches_per_epoch = {
  "mul.dis.all": 78801,
  "deu.rst.pcc": 541,
  "eng.pdtb.pdtb": 3000, // real: 10980
  "eng.rst.gum": 1700, // real: 3475
  "eng.rst.rstdt": 2000, // real: 4001
  "eng.sdrt.stac": 1200, // real: 2395
  "eus.rst.ert": 634,
  "fas.rst.prstc": 1025,
  "fra.sdrt.annodis": 547,
  "nld.rst.nldt": 402,
  "por.rst.cstn": 1037,
  "rus.rst.rrt": 2500, // real: 7217
  "spa.rst.rststb": 560,
  "spa.rst.sctb": 110,
  "tur.pdtb.tdb": 613,
  "zho.pdtb.cdtb": 915,
  "zho.rst.sctb": 110,
  // newly added corpora
  "eng.pdtb.tedm": 3000, // adapted from eng.pdtb.pdtb
  "eng.rst.gentle": 1700, // adapted from eng.rst.gum
  "tur.pdtb.tedm": 613, // adapted from tur.pdtb.tdb
  "zho.rst.gcdt": 2151, // real: 6454
  "eng.dep.covdtb": 2020, // real: 6060
  "eng.dep.scidtb": 2020, // real: 6060
  "ita.pdtb.luna": 955, 
  "por.pdtb.crpc": 2932, // real: 8797
  "por.pdtb.tedm": 2932, // real: 8797
  "tha.pdtb.tdtb": 2759, // real: 8278
  "zho.dep.scidtb": 802,
  "eng.pdtb.gdtb": 3000, // adapted from eng.pdtb.pdtb
};

{
  "dataset_reader" : {
    "type": "disrpt_2021_rel_flair_clone",
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": transformer_model_name,
        "max_length": 511
      }
    },
    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": transformer_model_name
    },
    "features": features
  },
  "train_data_path": std.extVar("TRAIN_DATA_PATH"),
  "validation_data_path": std.extVar("VALIDATION_DATA_PATH"),
  "model": {
    "type": "disrpt_2021_flair_clone",
    "embedder": {
      "type": "featureful_bert",
      "model_name": transformer_model_name,
      "max_length": 511,
      "train_parameters": true,
      "last_layer_only": true
    },
    "seq2vec_encoder": {
        "type": "bert_pooler",
        "pretrained_model": transformer_model_name
    },
    "feature_dropout": 0.0,
    "features": features,
  },
  "data_loader": {
    "batches_per_epoch": batches_per_epoch[corpus_name],
    // NOTE: if you need to change batch size, scale batches_per_epoch, which assumes
    // a batch size of 4, by an appropriate amount. E.g., if you need to make batch
    // size 2, then use `batches_per_epoch[corpus_name] * 2`
    "batch_size": 4, 
    "shuffle": true
  },
  "trainer": {
    "num_epochs": 100,
    "patience": 12,
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 2e-5,
      #"weight_decay": 0.05,
      #"betas": [0.9, 0.99],
      #"parameter_groups": [
      #  [[".*embedder.*transformer.*"], {"lr": 2e-5}]
      #],
    },
    #"learning_rate_scheduler": {
    #  "type": "slanted_triangular",
    #  "num_epochs": 50,
    #  "cut_frac": 0.1,
    #},
    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "factor": 0.6,
      "mode": "max",
      "patience": 2,
      "verbose": true,
      "min_lr": 5e-7
    },
    //"learning_rate_scheduler": {
    //  "type": "cosine",
    //  "t_initial": 5,
    //},
    "validation_metric": "+relation_accuracy"
  }
}

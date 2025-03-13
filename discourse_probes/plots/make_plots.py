import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import re
from plotly.subplots import make_subplots
import torch
import numpy as np


def load_results(path, framework=None, test_only=True):
    # each experiment folder contains multiple tables in subdirectories
    # load all of them
    tables = {}
    for root, dirs, files in os.walk(path):
        if framework is not None and root.find(f"framework={framework}") == -1:
            continue
        if root.find("old") != -1 or root.find("seed") == -1:
            # these are files from previous runs we don't need
            continue
        if test_only and root.find("test") == -1:
            continue
        for file in files:
            if file.endswith(".csv"):
                table = pd.read_csv(os.path.join(root, file))
                tables[os.path.join(root, file)] = table
    return tables

def load_all_results(path):
    tables = {}
    for dirname in os.listdir(path):
        dir_path = os.path.join(path, dirname)
        print(dir_path)
        if dir_path.find("mlp/") != -1:
            continue
        if os.path.isdir(dir_path) and \
            os.path.exists(os.path.join(dir_path, "results/")) and \
            dirname.find("rel_embeddings_") != -1:
            model_name = dirname.split("rel_embeddings_")[1]
            tables[model_name] = load_results(os.path.join(path, dirname))
    return tables

def get_overall_acc(results, experiment="unified", train_lang="all", model="mlp", score="overall_dataset_acc"):

    path_regex = f"{experiment}/train_lang={train_lang}/{model}/all_layers/final_full_train_set_training/test/general_stats.csv"
    path = None
    accuracies = []
    for p in results.keys():
        if p.find(path_regex) != -1 and p.find("seed") != -1:
            path = p
            df = results[path]
            accuracy = df[df["Statistic"] == score]["Value"].values[0]
            accuracies.append(float(accuracy))

    return accuracies

def get_layer_wise_acc(results, model="aya23_35b"):

    path_regex = "layer-wise/"
    fname = "general_stats.csv"

    paths = []
    accuracy = []
    pooling_type = []
    layer_id = []
    model_name = []
    for p in results.keys():
        if p.find(path_regex) != -1 and p.find(model) != -1 and p.find(fname) != -1 and p.find("seed") != -1:
            path = p
            paths.append(path)
            layer = int(re.findall(r"layer_([0-9]+)", path)[0])
            pooling = re.findall(r"pooling=([a-z\_]+)", path)[0]
            df = results[path]
            acc = df[df["Statistic"] == "accuracy"]["Value"].values[0]
            accuracy.append(acc)
            pooling_type.append(pooling)
            layer_id.append(layer)
            model_name.append(model)

    df = pd.DataFrame({"path": paths, "model_name": model_name, "accuracy": accuracy, "pooling": pooling_type, "layer": layer_id})

    df["accuracy"] = df["accuracy"].apply(lambda x: float(x))

    df_grouped = df.groupby(["layer", "pooling"]).agg(accuracy=("accuracy", "mean"), accuracy_var=("accuracy", "var")).reset_index()

    return df_grouped


def get_layer_wise_lang_acc_scores(results, model="aya23_35b"):

    path_regex = "layer-wise/"
    fname = "lang_acc.csv"

    paths = []
    accuracy = []
    pooling_type = []
    layer_id = []
    model_name = []
    langs = []

    for p in results.keys():
        if p.find(path_regex) != -1 and p.find(model) != -1 and p.find(fname) != -1 and p.find("dev_train_part") != -1 \
            and "final_full_train_set" not in p:
            layer = int(re.findall(r"layer=([0-9]+)", p)[0])
            pooling = re.findall(r"pooling=([a-z\_]+)", p)[0]
            df = results[p]

            for lang in df["language"].unique():
                acc = df[df["language"] == lang]["correct"].values[0]
                langs.append(lang)
                pooling_type.append(pooling)
                layer_id.append(layer)
                model_name.append(model)
                accuracy.append(acc)
                paths.append(p)

    return pd.DataFrame({"path": paths, 
                         "language": langs, 
                         "model_name": model_name, 
                         "accuracy": accuracy, 
                         "pooling": pooling_type, 
                         "layer": layer_id})


def make_model_vs_acc_plot(accuracies, suffix="", model2params=None, names2models=None):
    inclusive_colors = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
    # remove emptey values
    accuracies = {k: v for k, v in accuracies.items() if len(v) > 0}
    model_family_color_mapping = {
        "qwen": inclusive_colors[0],
        "aya": inclusive_colors[1],
        "llama": inclusive_colors[5],
        "bloom": inclusive_colors[3],
        "emma": inclusive_colors[4],
        "mistral": inclusive_colors[7],
        "phi": inclusive_colors[6],
    }
    models = [names2models[model_name].split("/")[-1] for model_name in accuracies.keys()]
    models = [model if model.find("Qwen") == -1 else model.replace("Qwen2.5", "Qwen2") for model in models]
    models = [model if model.find("Llama") == -1 else model.replace("Llama-3.1", "Llama3") for model in models]
    models = [model if model.find("Llama") == -1 else model.replace("Llama-3.2", "Llama3") for model in models]
    models = [model if model.find("aya") == -1 else model.replace("aya-23", "Aya23") for model in models]
    models = [model if model.find("aya") == -1 else model.replace("aya-expanse", "AyaExp") for model in models]
    models = [model if model.find("Mistral") == -1 else model.replace("Mistral-Small-24B-Base-2501", "Mistral-24B-Base") for model in models]
    models = [model if model.find("Bloom-560m") == -1 else model.replace("560m", "560M") for model in models]
    models = [model if model.find("emma") == -1 else model.replace("emma-500-llama2-7b", "Emma500-7B") for model in models]
    models = [model if model.find("Qwen") == -1 else model.replace("Qwen2-14B", "Qwen2-14B") for model in models]
    models = [model.replace("b", "B") for model in models]
    accuracy_means = [np.mean(accuracies[model_name]) for model_name in accuracies.keys()]
    accuracy_vars = [np.var(accuracies[model_name]) for model_name in accuracies.keys()]
    print(accuracy_means)
    print(accuracy_vars)
    print("hello1")
    df = pd.DataFrame({"model": models, "accuracy": accuracy_means, "accuracy_var": accuracy_vars,
                    "#params": [model2params[model_name] for model_name in accuracies.keys()],
                    "model family": [re.findall(r"^[a-zA-Z]+", model_name)[0].strip("z") for model_name in accuracies.keys()]}).sort_values("#params")
    
    df["accuracy"] = df["accuracy"] * 100
    df["accuracy_err"] = df["accuracy_var"].apply(lambda x: np.sqrt(x * 10000))
    
    df["color"] = [model_family_color_mapping[model_family] for model_family in df["model family"]]
    df = df.dropna(axis=0)
    # fig = px.scatter(df, x="#params", y="accuracy", text="model", color="model family", size_max=60 )
    def set_text_position(name):
        print(name)
        if name == 'Llama3-70B':
            print('hello')
            return 'middle left'
        if name == 'Llama3-8B':
            print('hello')
            return 'middle right'
        elif name == 'Qwen2-72B':
            print('hello')
            return 'top center'
        elif name == 'Qwen2-32B':
            print('hello')
            return 'bottom center'
        elif name == "Mistral-24B-Base":
            return 'middle right'
        #elif name == "Aya23-8B":
        #     return 'middle right'
        elif name == "Qwen2-0.5B":
            return 'middle right'
        elif name == "Qwen2-1.5B":
            return 'middle right'
        elif name == "Bloom-560m":
            return 'top center'
        elif name == "Bloom-1B1":
            return 'middle right'
        else:
            return 'middle left'
    # textposition = list(map(set_text_position, df['model']))
    df["textposition"] = df["model"].apply(set_text_position)
    print("hello2")
    df["color"] = [model_family_color_mapping[model_family] for model_family in df["model family"]]
    print(df.head())
    data = []
    for i, row in df.iterrows():
        x = row["#params"]
        y = row["accuracy"]
        y_var = row["accuracy_var"]
        error_y = dict(type='data', array=[row["accuracy_err"]], visible=True, thickness=1)
        text = row["model"]
        textposition = row["textposition"]
        print(row["model"])
        color = row["color"]
        print(x, y, text, color, error_y, textposition)
        data.append(go.Scatter(x=[x], y=[y], mode='markers+text', text=text,
                            error_y=error_y,
                            textposition=textposition,
                            textfont_size=12,
                            name=row["model family"],
                            showlegend=False,
                            marker=dict(size=8, color=color,
                            )))
    
    fig = go.Figure(data=data)
    fig.update_layout(xaxis_title="<b>number of parameters", yaxis_title="<b>average dataset accuracy (%)")
    # show error bars
    fig.update_xaxes(tickfont=dict(size=12))
    fig.update_yaxes(tickfont=dict(size=12))
    fig.update_layout(font=dict(size=12))
    # fig.update_traces(textposition=textposition, textfont_size=9)
    # log scale
    fig.update_xaxes(type="log")
    # make markers bigger
    # fig.update_traces(marker=dict(size=4))
    # make text smaller and move it to the right
    # fig.update_traces(textposition='middle right', textfont_size=9)
    # make background white, make lines grey
    fig.update_layout(plot_bgcolor='white',)
    # make grid lines grey
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', showline=True, linewidth=1, linecolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', showline=True, linewidth=1, linecolor='LightGray')

    # include red line to show reference
    fig.add_shape(type="line",
        x0=400000000, y0=47.9, x1=90000000000, y1=47.9,
        line=dict(color="red",width=1.0, dash="dot")
)

    fig.write_image(f"results/disco_probe_results/params_vs_accuracy{suffix}.png", scale=4, width=800, height=500)
    fig.write_image(f"results/disco_probe_results/params_vs_accuracy{suffix}.pdf", scale=4, width=800, height=500)
    fig.show()

def get_model_dataset_scores(results, train_lang="all", split="test", model="mlp"):

    path_regex = f"unified/train_lang={train_lang}/{model}/all_layers/final_full_train_set_training/{split}_all/dataset_acc.csv"

    paths = []
    accuracy = []
    model_names = []
    datasets = []

    for model_name in results.keys():
        for path in results[model_name].keys():
            if path.find(path_regex) != -1 and path.find("old") == -1 and path.find("seed") != -1:

                df = results[model_name][path]

                for ds in df["dataset"].unique():
                    acc = df[df["dataset"] == ds]["correct"].values[0]
                    datasets.append(ds)
                    accuracy.append(acc)
                    paths.append(path)
                    model_names.append(model_name)
    
    df = pd.DataFrame({"path": paths, "model_name": model_names, "dataset": datasets, "accuracy": accuracy})

    df["accuracy"] = df["accuracy"].apply(lambda x: float(x)) * 100

    df_grouped = df.groupby(["dataset", "model_name"]).agg(accuracy=("accuracy", "mean"), accuracy_var=("accuracy", "var")).reset_index()

    return df_grouped


def get_model_lang_scores(results):

    path_regex = "unified/framework=all_lang=all/dev_train_part/lang_acc.csv"

    paths = []
    accuracy = []
    model_names = []
    languages = []
    for model_name in results.keys():
        for path in results[model_name].keys():
            if path.find(path_regex) != -1:

                df = results[model_name][path]

                for ds in df["language"].unique():
                    acc = df[df["language"] == ds]["correct"].values[0]
                    languages.append(ds)
                    accuracy.append(acc)
                    paths.append(path)
                    model_names.append(model_name)
    
    return pd.DataFrame({"path": paths, "model_name": model_names, "language": languages, "accuracy": accuracy})


def get_model_lang_heatmap_scores(results, langs, model="mlp"):

    paths = []
    accuracy = []
    train_languages = []
    test_languages = []
    for lname_train, lang in langs.items():
        for lname_test, lang_test in langs.items():
            for path in results.keys():
                if path.find(f"unified/train_lang={lname_train}/{model}/all_layers/final_full_train_set_training/test_{lname_test}/general_stats.csv") != -1:
                    df = results[path]
                    acc = df[df["Statistic"] == "overall_dataset_acc"]["Value"].values[0]
                    train_languages.append(lname_train)
                    test_languages.append(lname_test)
                    accuracy.append(acc)
                    paths.append(path)

    
    return pd.DataFrame({"path": paths, "train_language": train_languages, "test_language": test_languages, "accuracy": accuracy})

def get_layer_wise_lang_acc_scores(results, model="aya23_35b", pooling="max", probe="mlp", split="test"):

    path_regex = f"layer-wise/pooling={pooling}/{probe}/"
    fname = "/general_stats.csv"

    paths = []
    accuracy = []
    layer_id = []
    langs = []

    for p in results.keys():
        if p.find(path_regex) != -1 and p.find("seed") != -1 and p.find(model) != -1 and p.find(fname) != -1 and p.find(f"/{split}_") != -1 and p.find("final_full_train_set") != -1:
            print(p)
            layer = int(re.findall(r"/layer_([0-9]+)", p)[0])
            df = results[p]
            lang = re.findall("/"+ split + r"_([a-z\-]+)", p)[0]
            acc = float(df[df["Statistic"] == "accuracy"]["Value"].values[0])
            
            accuracy.append(acc)
            layer_id.append(layer)
            langs.append(lang)
            paths.append(p)
    
    df = pd.DataFrame({"path": paths, "language": langs, "accuracy": accuracy, "layer": layer_id})

    df_grouped = df.groupby(["layer", "language"]).agg(accuracy=("accuracy", "mean"), accuracy_var=("accuracy", "var")).reset_index()

    return df_grouped

def get_layer_wise_lab_acc_scores(results, model="aya23_35b", pooling="max", probe="mlp", split="test"):

    path_regex = f"layer-wise/pooling={pooling}/{probe}/"
    fname = "/lab_acc.csv"

    paths = []
    accuracy = []
    layer_id = []
    labs = []

    for p in results.keys():
        if p.find("seed") != -1 and p.find(path_regex) != -1 and p.find(model) != -1 and p.find(fname) != -1 and p.find(f"/{split}/") != -1 and p.find("final_full_train_set") != -1:
            print(p)
            layer = int(re.findall(r"/layer_([0-9]+)", p)[0])
            df = results[p]

            for lab in df["label"].unique():
                acc = df[df["label"] == lab]["correct"].values[0]
                labs.append(lab)
                accuracy.append(acc)
                layer_id.append(layer)
                paths.append(p)

    df = pd.DataFrame({"path": paths, 
                         "label": labs,
                         "accuracy": accuracy, 
                         "layer": layer_id})
    
    df["accuracy"] = df["accuracy"].apply(lambda x: float(x))

    df_grouped = df.groupby(["layer", "label"]).agg(accuracy=("accuracy", "mean"), accuracy_var=("accuracy", "var")).reset_index()

    return df_grouped
                                                    


results = load_all_results("data/disrpt_private/")

model_df = pd.read_csv("data/disrpt_private/discoProb_models.csv")
model_df["param_count"] = model_df["No. of Params"].apply(lambda x: int(float(x.replace("B", "e9"))))

names2models = {'qwen25_0B5': "Qwen/Qwen2.5-0.5B",
                    'qwen25_1B5': "Qwen/Qwen2.5-1.5B",
                    'aya_exp_8b': "CohereForAI/aya-expanse-8b",
                    'aya_exp_32b': "CohereForAI/aya-expanse-32b",
                    'llama3_1B': "meta-llama/Llama-3.2-1B",
                    'bloomz_7b1': "bigscience/bloomz-7b1",
                    'bloom_1b': "bigscience/bloom-1b1",
                    'llama3_3B': "meta-llama/Llama-3.2-3B",
                    'llama3_70B': "meta-llama/Llama-3.1-70B",
                    'qwen25_32B': "Qwen/Qwen2.5-32B",
                    'qwen25_7B': "Qwen/Qwen2.5-7B",
                    'emma500': "MaLA-LM/emma-500-llama2-7b",
                    'qwen_25_14B': "Qwen/Qwen2.5-14B", 
                    'aya23_8b': "CohereForAI/aya-23-8B",
                    'qwen25_3B': "Qwen/Qwen2.5-3B",
                    'bloom_3b': "bigscience/bloom-3b",
                    'llama3_8B': "meta-llama/Llama-3.1-8B",
                    'aya23_35b': "CohereForAI/aya-23-35B",
                    'bloom_560m': "bigscience/bloom-560m",
                    'qwen25_14B': "Qwen/Qwen2.5-14B",
                    'bloom_7b1': "bigscience/bloom-7b1",
                    'qwen25_72B': "Qwen/Qwen2.5-72B",
                    'mistral_small_24B': "mistralai/Mistral-Small-24B-Base-2501",
                    'phi_4': "microsoft/phi-4",}

for name in names2models.keys():
    if names2models[name] not in model_df["Model Name"].tolist():
        print(name, "|"+names2models[name]+"|")

model2params = {model: model_df[model_df["Model Name"] == names2models[model]]["param_count"].values[0] for model in names2models.keys()}


accuracies = dict([(model_name, get_overall_acc(results[model_name])) for model_name in results.keys()])
print(accuracies)
make_model_vs_acc_plot(accuracies, suffix="_mlp_test", model2params=model2params, names2models=names2models)


# get table for aya23
languages = ["all", "eng", "nld", "eus", "fas", "por", "spa", "rus", "deu", "ita", "zho", "fra", "tha", "tur"]
languages_dict = dict((l, l) for l in languages)
groups = {"germanic": ["eng", "nld", "deu"], "romance": ["por", "spa", "ita", "fra"], "indo-european": ["eng", "nld", "fas", "por", "spa", "rus", "deu", "ita", "fra"]}

# combine into one dict
all_langs = languages_dict | groups

df_mlp = get_model_lang_heatmap_scores(results["aya23_35b"], all_langs).sort_values(["test_language", "train_language"])
# df_linear = get_model_lang_heatmap_scores(results["aya23_35b"], all_langs, model="linear").sort_values(["test_language", "train_language"])

baselines = {"all": 0.6566, "eng": 0.6933,
             "nld": 0.5662, "eus": 0.6239, "fas": 0.5270, "por": 0.7012, "spa": 0.6354, "rus": 0.6465, "deu": 0.4654, 
             "ita": 0.5211, "zho": 0.7170, "fra": 0.5616, "tha": 0.8430, "tur": 0.6101, 
             "germanic": 0.6539, "romance": 0.6367, "indo-european": 0.6464}

baseline_all = {"all": 0.4791, "eng": 0.5289,
             "nld": 0.3723, "eus": 0.3599, "fas": 0.4595, "por": 0.5293, "spa": 0.4428, "rus": 0.5058, "deu": 0.25, 
             "ita": 0.4711, "zho": 0.5033, "fra": 0.2656, "tha": 0.9055, "tur": 0.3387, 
             "germanic": 0.4805, "romance": 0.4586, "indo-european": 0.4730}

add_rows = []
for lang in all_langs.keys():
    if lang in baselines:
        add_rows.append({"test_language": lang, "train_language": "DisCoDisCo (dataset-wise)", "accuracy": baselines[lang]})
        add_rows.append({"test_language": lang, "train_language": "DisCoDisCo (all)", "accuracy": baseline_all[lang]})
df_mlp = pd.concat([df_mlp, pd.DataFrame(add_rows)], ignore_index=True)

order = ["deu", "eng", "eus", "fas", "fra", "ita", "nld", "por", "rus", "spa", "tha", "tur", "zho", "germanic", "romance", "indo-european", "all", "DisCoDisCo (all)", "DisCoDisCo (dataset-wise)"]

mapping = {"zho": "Mand. Chinese", "eus": "Basque", "tur": "Turkish", "tha": "Thai", "romance": "Romance", "germanic": "Germanic", "indo-european": "Indo-European", "all": "all", "eng": "English", "nld": "Dutch", "fas": "Farsi", "por": "Portuguese", "spa": "Spanish", "rus": "Russian", "deu": "German", "ita": "Italian", "fra": "French"}

df_mlp["order_key1"] = df_mlp["test_language"].apply(lambda x: order.index(x))
df_mlp["order_key2"] = df_mlp["train_language"].apply(lambda x: -order.index(x))
df_mlp = df_mlp.sort_values(["order_key1", "order_key2"]).sort_values(["order_key1"])

df_mlp["test_language"] = df_mlp["test_language"].apply(lambda x: mapping[x] if x in mapping else x)
df_mlp["train_language"] = df_mlp["train_language"].apply(lambda x: mapping[x] if x in mapping else x)

df_mlp["accuracy"] = df_mlp["accuracy"].apply(lambda x: float(x))

# calculate mean and error for each entry
df_mlp_grouped = df_mlp.groupby(["test_language", "train_language"]).agg(accuracy=("accuracy", "mean"), accuracy_var=("accuracy", "var"), order_key1=("order_key1", "first"), order_key2=("order_key2", "first")
                                                                         ).reset_index().sort_values(["order_key1", "order_key2"])

df_mlp_grouped["percentage"] = df_mlp_grouped["accuracy"] * 100

# show the values of the heatmap
fig = go.Figure(data=go.Heatmap(
                        z=df_mlp_grouped["percentage"],
                        x=df_mlp_grouped["test_language"],
                        y=df_mlp_grouped["train_language"],
                        colorscale=[(0, "#0072B2"), (0.5, "#F0E442"), (1, "#D55E00")],
                        zmin=0.0,
                        zmax=100.0,
                        colorbar=dict(title="<b>accuracy (%)"),
                        text=df_mlp_grouped["percentage"],
                        texttemplate="%{text:.1f}",
                        ))
fig.update_layout(
    xaxis_title="<b>test partition",
    yaxis_title="<b>train partition",
)
fig.update_layout(font=dict(color='black', size=14,)) 
fig.update_xaxes(tickfont=dict(color='black', size=12))
fig.update_yaxes(tickfont=dict(color='black', size=12))
fig.update_xaxes(tickangle=20)

fig.update_coloraxes(colorbar=dict(title="<b>accuracy (%)", tickfont=dict(color='black', size=14)))

# move axis titles closer to the plot
fig.update_xaxes(title_standoff=0)
fig.update_yaxes(title_standoff=0)

fig.add_shape(type="line",
    x0=-0.5, y0=1.5, x1=16.5, y1=1.5,
    line=dict(color="White",width=2.5)
)

fig.add_shape(type="line",
    x0=-0.5, y0=5.5, x1=16.5, y1=5.5,
    line=dict(color="White",width=2.5)
)

fig.add_shape(type="line",
    x0=12.5, y0=-0.5, x1=12.5, y1=18.5,
    line=dict(color="White",width=2.5)
)

fig.write_image("results/disco_probe_results/lang_heatmap_mlp.png", scale=4, width=1000, height=480)
fig.write_image("results/disco_probe_results/lang_heatmap_mlp.pdf", scale=4, width=1000, height=480)
fig.show()

# get layer-wise accuracy plot by lang
inclusive_colors = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
lang_color_dict = {
    "zho": inclusive_colors[1],
    "eus": inclusive_colors[2],
    "tur": inclusive_colors[3],
    "tha": inclusive_colors[4],
    "romance": inclusive_colors[5],
    "germanic": inclusive_colors[6],
    "indo-european": inclusive_colors[7]
}

mapping = {"zho": "Mandarin Chinese", "eus": "Basque", "tur": "Turkish", "tha": "Thai", "romance": "Romance", "germanic": "Germanic", "indo-european": "Indo-European"}
order = ["all", "Basque", "Mandarin Chinese", "Thai", "Turkish", "Germanic", "Romance", "Indo-European"]
updated_color_dict = {mapping[k]: v for k, v in lang_color_dict.items()}

layer_wise_acc = get_layer_wise_lang_acc_scores(results["aya23_35b"], model="aya23_35b", pooling="max", probe="mlp", split="test")
layer_wise_acc["language"] = layer_wise_acc["language"].apply(lambda x: mapping[x] if x in mapping else x)
layer_wise_acc["layer"] = layer_wise_acc["layer"].astype(int) + 1

# filter to langs and sort by order and then by layer
layer_wise_acc = layer_wise_acc[layer_wise_acc["language"].isin(order)]
layer_wise_acc["order_key1"] = layer_wise_acc["language"].apply(lambda x: order.index(x))
layer_wise_acc["order_key2"] = layer_wise_acc["layer"]
layer_wise_acc["accuracy"] = layer_wise_acc["accuracy"].apply(lambda x: float(x) * 100)
layer_wise_acc["accuracy_err"] = layer_wise_acc["accuracy_var"].apply(lambda x: np.sqrt(x * 10000))
layer_wise_acc = layer_wise_acc.sort_values(["order_key1", "order_key2"])



data = []
for lang in layer_wise_acc["language"].unique():
    if lang in ["eng", "nld", "deu", "por", "spa", "ita", "fra", "rus", "fas"]:
        continue
    df = layer_wise_acc[layer_wise_acc["language"] == lang].sort_values("layer")
    if lang == "all":
        data.append(go.Scatter(x=df["layer"], y=df["accuracy"], error_y=dict(type='data', array=df['accuracy_err'], visible=True, thickness=1),
                               mode="lines+markers", name=lang, line=dict(color="#000000", width=4), marker=dict(size=1)))
    else:
        data.append(go.Scatter(x=df["layer"], y=df["accuracy"], error_y=dict(type='data', array=df['accuracy_err'], visible=True, thickness=1),
                               mode="lines+markers", name=lang, 
                               line=dict(color=updated_color_dict[lang], width=2), marker=dict(size=1)))

fig = go.Figure(data=data)
# add legend
fig.update_layout(legend=dict(
    yanchor="top",
    y=1.2,
    xanchor="right",
    x=0.95,
))
fig.update_layout(legend_orientation="h")
fig.update_layout(font=dict(color='black', size=17,)) 
fig.update_xaxes(tickfont=dict(color='black', size=17))
fig.update_yaxes(tickfont=dict(color='black', size=17))

fig.update_xaxes(range=[0, 41])
fig.update_layout(xaxis_title="<b>layer", yaxis_title="<b>accuracy (%)")
# change background color
fig.update_layout(plot_bgcolor='white',)
# make grid lines grey
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', showline=True, linewidth=1, linecolor='LightGray')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', showline=True, linewidth=1, linecolor='LightGray')
fig.write_image("results/disco_probe_results/layer_wise_acc_lang.png", scale=5, width=800, height=550)
fig.write_image("results/disco_probe_results/layer_wise_acc_lang.pdf", scale=5, width=800, height=550)


layer_wise_acc = get_layer_wise_acc(results["aya23_35b"]).sort_values("layer")
fig = px.line(layer_wise_acc, x="layer", y="accuracy", color="pooling")
fig.update_layout(plot_bgcolor='white',)
# make grid lines grey
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', showline=True, linewidth=1, linecolor='LightGray')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', showline=True, linewidth=1, linecolor='LightGray')
fig.write_image("results/disco_probe_results/layer_wise_acc.png")



layer_wise_acc = get_layer_wise_lab_acc_scores(results["aya23_35b"], model="aya23_35b", pooling="max", probe="mlp", split="test")


label_groups = {'elaboration': 'thematic',
                'structuring': 'structuring',
                'topic-change': 'topic-management',
                'adversative': 'causal-argumentative',
                'causal': 'causal-argumentative',
                'temporal': 'temporal',
                'framing': 'thematic',
                'evaluation': 'causal-argumentative',
                'enablement': 'causal-argumentative',
                'attribution': 'thematic',
                'topic-comment': 'topic-management',
                'explanation': 'causal-argumentative',
                'contingency': 'causal-argumentative',
                'mode': 'thematic',
                'comparison': 'thematic',
                'reformulation': 'thematic',
                'topic-adjustment': 'topic-management'}
layer_wise_acc["label_group"] = layer_wise_acc["label"].apply(lambda x: label_groups[x] if x in label_groups else "other")
layer_wise_acc["layer"] = layer_wise_acc["layer"].astype(int) + 1
layer_wise_acc["accuracy"] = layer_wise_acc["accuracy"].apply(lambda x: float(x) * 100)
layer_wise_acc["accuracy_err"] = layer_wise_acc["accuracy_var"].apply(lambda x: np.sqrt(x * 10000))

# fig = make_subplots(rows=2, cols=3, subplot_titles=layer_wise_acc["label_group"].unique(), vertical_spacing=0.15, horizontal_spacing=0.05)
for j, lab_group in enumerate(layer_wise_acc["label_group"].unique()):
    data = []
    for i, lab in enumerate(layer_wise_acc.loc[layer_wise_acc["label_group"] == lab_group, "label"].unique()):
        color = inclusive_colors[i]
        df = layer_wise_acc[layer_wise_acc["label"] == lab].sort_values("layer")
        # fig.add_trace(go.Scatter(x=df["layer"], y=df["accuracy"], mode="lines+markers", name=lab, line=dict(width=2, color=color), marker=dict(size=1), legendgroup=j), row=row, col=col)
        
        data.append(go.Scatter(x=df["layer"], y=df["accuracy"], error_y=dict(type='data', array=df['accuracy_err'], visible=True, thickness=1),
                               mode="lines", name=lab, line=dict(width=1, color=color), marker=dict(size=1)))
        # let plots start at 0
    fig = go.Figure(data=data)
    fig.update_yaxes(range=[0, 75])
    fig.update_xaxes(range=[0, 41])
    if lab_group == "thematic":
        fig.update_layout(legend=dict(
            yanchor="top",
            y=1.2,
            xanchor="right",
            x=1.11,
        ))
    elif lab_group == "causal-argumentative":
        fig.update_layout(legend=dict(
            yanchor="top",
            y=1.1,
            xanchor="right",
            x=1.01,
        ))
    else:
        fig.update_layout(legend=dict(
            yanchor="top",
            y=1.1,
            xanchor="right",
            x=1.1,
        ))

    # mage legend go horizontally
    fig.update_layout(legend_orientation="h")
    # restrict maximum legend width
    fig.update_yaxes(title_text="<b>accuracy (%)")
    fig.update_xaxes(title_text="<b>layer")
    # make axis labels and titles black+bold
    fig.update_layout(font=dict(color='black', size=18,)) 
    fig.update_xaxes(tickfont=dict(color='black', size=18))
    fig.update_yaxes(tickfont=dict(color='black', size=18))

    fig.update_layout(plot_bgcolor='white',)
    # make grid lines grey
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', showline=True, linewidth=1, linecolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', showline=True, linewidth=1, linecolor='LightGray')
    # remove legend
    fig.write_image(f"results/disco_probe_results/layer_wise_lab_acc_{lab_group}.png", width=600, height=410, scale=5)
    fig.write_image(f"results/disco_probe_results/layer_wise_lab_acc_{lab_group}.pdf", width=600, height=410, scale=5)


# make this plot for accumulated groups
layer_wise_acc = get_layer_wise_lab_acc_scores(results["aya23_35b"], model="aya23_35b", pooling="max", probe="mlp", split="test")
layer_wise_acc["accuracy"] = layer_wise_acc["accuracy"].apply(lambda x: float(x) * 100)
layer_wise_acc["accuracy_err"] = layer_wise_acc["accuracy_var"].apply(lambda x: np.sqrt(x * 10000))
layer_wise_acc["label_group"] = layer_wise_acc["label"].apply(lambda x: label_groups[x] if x in label_groups else "other")
layer_wise_acc["layer"] = layer_wise_acc["layer"].astype(int) + 1
grouped_acc = layer_wise_acc.groupby(["layer", "label_group"]).agg(accuracy=("accuracy", "mean"), accuracy_err=("accuracy_err", "mean")).reset_index()

data = []
for i, lab_group in enumerate(grouped_acc["label_group"].unique()):
    color = inclusive_colors[i]
    df = grouped_acc[grouped_acc["label_group"] == lab_group].sort_values("layer")
    data.append(go.Scatter(x=df["layer"], y=df["accuracy"], error_y=dict(type='data', array=df['accuracy_err'], visible=True, thickness=1),
                            mode="lines", name=lab_group, line=dict(width=1, color=color), marker=dict(size=1)))

fig = go.Figure(data=data)
fig.update_yaxes(range=[0, 70])
fig.update_xaxes(range=[0, 41])
fig.update_layout(legend=dict(
    yanchor="top",
    y=1.1,
    xanchor="right",
    x=1.07
))
# mage legend go horizontally
fig.update_layout(legend_orientation="h")
fig.update_yaxes(title_text="<b>accuracy (%)")
fig.update_xaxes(title_text="<b>layer")
# make axis labels and titles black+bold
fig.update_layout(font=dict(color='black', size=18,))
fig.update_xaxes(tickfont=dict(color='black', size=18))
fig.update_yaxes(tickfont=dict(color='black', size=18))
fig.update_layout(plot_bgcolor='white',)
# make grid lines grey
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', showline=True, linewidth=1, linecolor='LightGray')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', showline=True, linewidth=1, linecolor='LightGray')

fig.write_image("results/disco_probe_results/layer_wise_lab_acc_grouped.png", width=600, height=410, scale=5)
fig.write_image("results/disco_probe_results/layer_wise_lab_acc_grouped.pdf", width=600, height=410, scale=5)


# dataset scores
dataset_scores = get_model_dataset_scores(results, train_lang="all", split="test", model="mlp")
dataset_scores["accuracy"] = dataset_scores["accuracy"].apply(lambda x: float(x) * 100)
dataset_scores["accuracy_err"] = dataset_scores["accuracy_var"].apply(lambda x: np.sqrt(x * 10000))

dataset_scores["model_name"] = dataset_scores["model_name"].apply(lambda x: names2models[x].split("/")[1])
dataset_scores = dataset_scores.sort_values(["dataset", "model_name"])
dataset_scores["model_name"] = dataset_scores["model_name"].apply(lambda x: "\\rot{90}{"+x+"}")
dataset_scores["accuracy"] = dataset_scores["accuracy"].astype(float)
# add avg for each model
avg_scores = dataset_scores.groupby("model_name").agg({"accuracy": "mean"}).reset_index()
avg_scores["dataset"] = "average"
dataset_scores = pd.concat([dataset_scores, avg_scores], ignore_index=True)
# format scores to 4 decimal places
dataset_scores["accuracy"] = dataset_scores["accuracy"].apply(lambda x: str(round(x, 1)))

# dataset_scores.to_csv("results/disco_probe_results/dataset_scores.csv")
# make a table
df = dataset_scores.pivot(index="dataset", columns="model_name", values="accuracy")
df.to_csv("results/disco_probe_results/dataset_scores.csv")

# get test set predictions for aya23 35B and compute confusion matrix
preds = []

for seed in [14, 42, 123, 999, 5555]:
    preds.append(pd.read_csv(f"data/disrpt_private/rel_embeddings_aya23_35b/results/seed={seed}/store_full_preds/mlp/all_layers/final_full_train_set_training/test_all/test_set_results.csv"))

df_preds = preds[0]

df_preds["label"] = df_preds["label"].apply(lambda x: x.split("/")[-1])
counts = df_preds.groupby(["label", "prediction"]).size().reset_index(name="count")

confusion_matrix = counts.pivot(index="label", columns="prediction", values="count").fillna(0)
# make int
confusion_matrix = confusion_matrix.astype(int)

# normalize by row for color
confusion_matrix_color = confusion_matrix.div(confusion_matrix.sum(axis=1), axis=0)


fig = go.Figure(data=go.Heatmap(
                        z=confusion_matrix_color,
                        x=confusion_matrix.columns,
                        y=confusion_matrix.index,
                        colorscale=[(0, "#0072B2"), (0.5, "#F0E442"), (1, "#D55E00")],
                        colorbar=dict(title="<b>count"),
                        text=confusion_matrix,
                        texttemplate="%{text}",
                        hoverinfo="text",
                        ))

# show text
fig.update_layout(font=dict(color='black', size=16,))
fig.update_xaxes(tickfont=dict(color='black', size=16))
fig.update_yaxes(tickfont=dict(color='black', size=14))
# move axis labels closer to axis
fig.update_xaxes(title_standoff=0)
fig.update_yaxes(title_standoff=0)
fig.update_layout(xaxis_title="<b>predicted label", yaxis_title="<b>gold label")
fig.update_layout(plot_bgcolor='white',)
fig.update_xaxes(tickangle=-35)
fig.update_traces(showscale=False)

fig.write_image("results/disco_probe_results/confusion_matrix.png", scale=4, width=800, height=550)
fig.write_image("results/disco_probe_results/confusion_matrix.pdf", scale=4, width=800, height=550)

fig.show()

# get test set predictions for aya23 35B and compute confusion matrix
df_preds = results["aya23_35b"]["data/disrpt_private/rel_embeddings_aya23_35b/results/store_full_preds/mlp/all_layers/test_all/test_set_results.csv"]


# make probe weight plot

probe_weights = torch.load("data/disrpt_private/rel_embeddings_aya23_35b/results/store_full_preds/probe_weights_True_classifier.pt")
probe_weights.keys()

sums = probe_weights["1.weight"].abs().sum(dim=0)

# sum every 48 sums to get layer-wise sums
layer_sums = sums.view(40, -1).sum(dim=1).cpu().numpy()

fig = go.Figure(data=go.Scatter(x=list(range(1, 41)), y=layer_sums, mode="lines"))
fig.update_layout(font=dict(color='black', size=18,))
fig.update_xaxes(tickfont=dict(color='black', size=12))
fig.update_yaxes(tickfont=dict(color='black', size=12))
# show axis
# fig.update_layout(xaxis_color='black', yaxis_color='black', xaxis_linecolor='black', yaxis_linecolor='black')
fig.update_layout(xaxis_gridcolor='LightGray', yaxis_gridcolor='LightGray')
fig.update_layout(plot_bgcolor='white',)
fig.update_layout(xaxis_title="<b>layer", yaxis_title="<b>absolute weight sum")

fig.write_image("results/disco_probe_results/probe_first_layer_weight_sum.png", scale=5, width=800, height=600)
fig.write_image("results/disco_probe_results/probe_first_layer_weight_sum.pdf", scale=5, width=800, height=600)

prod = torch.matmul(probe_weights["5.weight"], probe_weights["1.weight"])
sums = prod.abs().sum(dim=0)
feature_sums = sums.view(40, -1).sum(dim=1).cpu().numpy()

fig = go.Figure(data=go.Scatter(x=list(range(1, 41)), y=feature_sums, mode="lines"))
fig.update_layout(font=dict(color='black', size=18,))
fig.update_xaxes(tickfont=dict(color='black', size=12))
fig.update_yaxes(tickfont=dict(color='black', size=12))
# show axis
# fig.update_layout(xaxis_color='black', yaxis_color='black', xaxis_linecolor='black', yaxis_linecolor='black')
fig.update_layout(xaxis_gridcolor='LightGray', yaxis_gridcolor='LightGray')
fig.update_layout(plot_bgcolor='white',)
fig.update_layout(xaxis_title="<b>layer", yaxis_title="<b>absolute weight sum")

fig.write_image("results/disco_probe_results/probe_mul_weight_sum.png", scale=5, width=800, height=600)
fig.write_image("results/disco_probe_results/probe_mul_weight_sum.pdf", scale=5, width=800, height=600)


data = []
prod = torch.matmul(probe_weights["5.weight"], probe_weights["1.weight"])

for i in range(prod.shape[0]):
    label_scores = prod[i].cpu().numpy()
    data.append(go.Scatter(x=list(range(1, 41)), y=label_scores, mode="lines", line_color=inclusive_colors[i // len(inclusive_colors)], name=f"label {i}"))

fig = go.Figure(data=data)
fig.update_layout(font=dict(color='black', size=18,))
fig.update_xaxes(tickfont=dict(color='black', size=12))
fig.update_yaxes(tickfont=dict(color='black', size=12))
# show axis
# fig.update_layout(xaxis_color='black', yaxis_color='black', xaxis_linecolor='black', yaxis_linecolor='black')
fig.update_layout(xaxis_gridcolor='LightGray', yaxis_gridcolor='LightGray')
fig.update_layout(plot_bgcolor='white',)
fig.update_layout(xaxis_title="<b>layer", yaxis_title="<b>absolute weight sum")

fig.write_image("results/disco_probe_results/probe_mul_weight_by_lab_sum.png", scale=5, width=800, height=600)
fig.write_image("results/disco_probe_results/probe_mul_weight_by_lab_sum.pdf", scale=5, width=800, height=600)
import os
import io
import re
import json
import shutil


def json2dict(json_file):
    return json.load(open(json_file))


def read_and_modify_rels_file(rels_file):

    updated_rels_lines = []

    filename = rels_file.split(os.sep)[-1]
    assert filename.endswith(".rels")
    dataset_name = filename.split("_")[0]

    lines = io.open(rels_file, "r", encoding="utf-8").read().strip().split("\n")
    header = lines[0]
    updated_rels_lines.append(header)

    for line in lines[1:]:
        if "\t" in line:
            fields = line.split("\t")
            fields_unchanged = fields[:-1]

            if "eng.rst.gum" not in rels_file:
                label_field = fields[-1]
            else:
                label_field = fields[-2]
            label_field_updated = unified_label_dict[label_field]

            line_updated = fields_unchanged + [label_field_updated]
            assert len(fields) == len(line_updated)

            updated_rels_lines.append("\t".join(line_updated))

    assert len(updated_rels_lines) == len(lines)

    modified_dataset_dir = f"{data_unified_dir}/{dataset_name}"
    os.makedirs(modified_dataset_dir, exist_ok=True)

    with open(os.path.join(modified_dataset_dir, filename), "w", encoding="utf-8") as modified_file:
        modified_file.write("\n".join(updated_rels_lines))

    print(f"o Done processing and updating {filename}!")


if __name__ == "__main__":
    # specify paths
    script_dir = os.path.dirname(__file__)
    data_dir = script_dir.replace("scripts", "data")
    data_unified_dir = script_dir.replace("scripts", "data_unified")
    out_dir = script_dir.replace("scripts", "output")  # contains the unified label .json files

    # obtain the unified label dict
    unified_label_json_filename = "alles_unified_labels.json"
    unified_label_dict = json2dict(f"{out_dir}/{unified_label_json_filename}")

    # test
    # read_and_modify_rels_file(os.path.join(os.path.join(data_dir, "eng.rst.gum"), "eng.rst.gum_train.rels"))

    for dataset_dir in os.listdir(data_dir):
        if re.match(r"[a-z]", dataset_dir):
            for file in os.listdir(f"{data_dir}/{dataset_dir}"):
                if file.endswith(".rels"):
                    read_and_modify_rels_file(f"{data_dir}/{dataset_dir}/{file}")
                elif file.endswith(".conllu"):
                    modified_dataset_dir = f"{data_unified_dir}/{dataset_dir}"
                    os.makedirs(modified_dataset_dir, exist_ok=True)
                    src_dir = f"{data_dir}/{dataset_dir}/{file}"
                    dst_dir = f"{data_unified_dir}/{dataset_dir}/{file}"
                    shutil.copy(src_dir, dst_dir)
                else:
                    pass

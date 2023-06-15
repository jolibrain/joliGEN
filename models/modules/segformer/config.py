import os
import json


def load_config_file(path):
    with open(path, "r") as f:
        print("path: ", path)
        temp = json.loads(f.read())
        if "base" in temp.keys():
            dir_path = os.path.dirname(path)
            new_path = os.path.join(dir_path, temp["base"])
            base_json = load_config_file(new_path)
            del temp["base"]
            base_json["backbone"].update(temp["backbone"])
            base_json["decode_head"].update(temp["decode_head"])
            temp = base_json

    return temp

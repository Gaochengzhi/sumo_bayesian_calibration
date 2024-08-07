import logging
import traceback
import os
import shutil
import json
import pandas as pd
import numpy


def round_dic_data(dic_data, decimal_precision=4):
    return {k: round(v, decimal_precision) for k, v in dic_data.items()}


def params_to_tuple(params):
    return tuple(sorted(params.items()))


def handle_exception(e):
    if not isinstance(e, numpy.linalg.LinAlgError):
        logging.error(e)
        logging.error(traceback.format_exc())


def copy_files(files, source_dir, destination_dir):
    for file in files:
        source_file = os.path.join(source_dir, file)
        destination_file = os.path.join(destination_dir, file)
        if os.path.exists(source_file):
            shutil.copy2(source_file, destination_file)
        else:
            print(f"warning {source_file} not exist")


def get_latest_file(folder="../log", suffix=".log"):
    log_files = [f for f in os.listdir(folder) if f.endswith(suffix)]
    if not log_files:
        print(f"No files with suffix '{suffix}' found in {folder}")
        return None

    latest_file = max(
        log_files, key=lambda f: os.path.getmtime(os.path.join(folder, f))
    )
    return os.path.join(folder, latest_file)


def json2pd(file_name=""):
    with open(file_name, "r") as file:
        log_content = file.read()

    data_list = [
        json.loads(entry) for entry in log_content.split("\n") if entry.strip()
    ]
    df = pd.DataFrame(
        [{**entry["params"], "target": entry["target"]} for entry in data_list]
    )
    return df

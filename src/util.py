import logging
import traceback
import os, shutil


def round_dic_data(dic_data, decimal_precision=2):
    return {k: round(v, decimal_precision) for k, v in dic_data.items()}


def handle_exception(e):
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

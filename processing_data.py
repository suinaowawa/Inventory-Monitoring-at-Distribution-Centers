'''Extract target data from txt file and filter out data with target label and save file name in JSON'''
import json
import logging
from typing import List

def extract_label(filename: str) -> str:
    """Extract Label from data's meta file

    Args:
        filename (str): meta data json file path

    Returns:
        str: quantity in image as label
    """
    with open(filename, 'r') as f:
        d=json.load(f)
    quantity = d['EXPECTED_QUANTITY']
    return quantity

def process_data(data_list: List[str], meta_dir: str, output_path: str) -> None:
    """Process data in data list, open it's meta file and check label, save image file name with
    target label in output json file

    Args:
        data_list (List[str]): List of to-be-processed data files' names
        meta_dir (str): directory where metadata files are saved
        output_path (str): output location to save the image files list json
    """
    # select image file with label 0-5 and save image file name in list
    selected_object = {0: [], 1:[], 2:[], 3:[], 4:[], 5:[]}
    for data in data_list:
        data = int(data)
        try:
            filename = f"{meta_dir}/{data:05d}.json"
            label = extract_label(filename)
            logging.info(f"{filename}:{label}")
            if int(label) < 6:
                selected_object[int(label)].append(f"{data:05d}.jpg")
                logging.info(f"Saving to {data:05d}.jpg")
        except:
            continue
    with open(output_path, "w") as f:
        json.dump(selected_object, f)


if __name__ == '__main__':
    with open(f"/opt/ml/processing/txtfile/random_val.txt", 'r') as f:
        lines = f.readlines()
        random_val = [line.strip() for line in lines]

    process_data(random_val, f"/opt/ml/processing/input", f"/opt/ml/processing/output/val.json")
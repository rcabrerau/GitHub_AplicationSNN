import h5py
import logging
import sys

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("log_estructuraH5.txt"),
                        logging.StreamHandler(sys.stdout)
                    ])

file_path = "E:/MASTER UOC/AULAS_4TO_SEMESTRE/TFM/AplicationSNN/datasetConvertion_CNN_to_SNN/SPIKING_labels/spiking_data_labels_22062024_1529_test.h5"

def inspect_h5_file(file_path):
    with h5py.File(file_path, 'r') as f:
        def print_structure(name, obj):
            logging.info(f"{name} {dict(obj.attrs)}")
            if isinstance(obj, h5py.Dataset):
                logging.info(f"Dataset: {name} Shape: {obj.shape} Dtype: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                logging.info(f"Group: {name}")
        f.visititems(print_structure)

inspect_h5_file(file_path)

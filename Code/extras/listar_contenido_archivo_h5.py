import h5py
import logging
import sys

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("log_read_spiking.txt"),
                        logging.StreamHandler(sys.stdout)
                    ])

file_path = "E:/MASTER UOC/AULAS_4TO_SEMESTRE/TFM/AplicationSNN/datasetConvertion_CNN_to_SNN/SPIKING_labels/spiking_data_labels_22062024_1529_test.h5"

def print_hdf5_structure(file_path):
    with h5py.File(file_path, 'r') as f:
        def print_attrs(name, obj):
            logging.info(name)
            for key, val in obj.attrs.items():
                logging.info(f"    {key}: {val}")
                
        f.visititems(print_attrs)

print_hdf5_structure(file_path)
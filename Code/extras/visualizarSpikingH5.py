import h5py
import torch
import logging
import sys

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("log_convert_spiking_test.txt"),
                        logging.StreamHandler(sys.stdout)
                    ])

def load_and_display_spiking_data(file_path, num_samples=3):
    with h5py.File(file_path, 'r') as f:
        keys = list(f.keys())
        sample_keys = keys[:num_samples]
        
        for key in sample_keys:
            spiking_data = torch.tensor(f[key]['spiking_data'][:])
            label = f[key].attrs['label']
            print(f"Sample: {key}, Label: {label}")
            print(spiking_data)

if __name__ == '__main__':
    file_path = "E:/MASTER UOC/AULAS_4TO_SEMESTRE/TFM/AplicationSNN/datasetConvertion_CNN_to_SNN/SPIKING_labels/spiking_data_labels_22062024_1529_test.h5"
    num_samples = 3  # Número de muestras a visualizar

    load_and_display_spiking_data(file_path, num_samples)

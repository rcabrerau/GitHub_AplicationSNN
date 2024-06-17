import torch

if torch.cuda.is_available():
    print("¡GPU disponible!")
else:
    print("No se encontró una GPU.")
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device')




from codecarbon import EmissionsTracker

def power_intensive_task():
    n = 1000**7
    return n * (n + 1) // 2


def track():
    # Start CodeCarbon tracker
    tracker = EmissionsTracker()

    try:
        # Start measuring emissions
        tracker.start()

        # Run power-intensive task
        result = power_intensive_task()
        print("Result:", result)

    finally:
        # Stop measuring emissions
        tracker.stop()

        # Print the emissions report
        print("Carbon Emissions:", tracker._emissions)

if __name__ == "__main__":
    track()





""" import pynvml
pynvml.nvmlInit()
device_count = pynvml.nvmlDeviceGetCount()
handle = pynvml.nvmlDeviceGetHandleByIndex(0) 
pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) """
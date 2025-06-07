import os
import sys
from pathlib import Path

# Print Python Path before changing the working directory
print("Before changing directory:")
print(sys.path)

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Print Python Path after changing the working directory
print("After changing directory:")
print(sys.path)

# Add the full path to retina explicitly
sys.path.append('/home/nicolharper/Documents/LukeCodeOrig/RetinalModel-main/retina')

print("After appending retina to sys.path:")
print(sys.path)

# Now try to import again
#from retina.analysis.latency import NoiseReconstructionDatasetBuilder


root = os.path.expanduser("~/Documents/LukeCodeOrig/RetinalModel-main")

sys.path.append('/home/nicolharper/Documents/LukeCodeOrig/RetinalModel-main/retina/analysis')
from latency import NoiseReconstructionDatasetBuilder
import torch

pred_offset = 128
length = 72


def build_and_save_reconstruction_dataset(istrain=True, noise=0):
    name = f"{pred_offset}_{length}_{noise}"
    output_path = f"{root}/data/recon/{name}"
    if not os.path.exists(output_path):
        Path(output_path).mkdir(parents=True, exist_ok=False)

    builder = NoiseReconstructionDatasetBuilder(root, pred_offset, istrain, noise, length)
    builder.build()

    torch.save(builder.x_frame, f"{output_path}/{istrain}_x_frame.pt")
    torch.save(builder.noise_frame, f"{output_path}/{istrain}_noise_frame.pt")
    torch.save(builder.x_frame_noise, f"{output_path}/{istrain}_x_frame_noise.pt")
    # torch.save(builder.rate_code, f"{output_path}/{istrain}_rate_code.pt")
    torch.save(builder.latency_code, f"{output_path}/{istrain}_latency_code.pt")


for noise in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]:
    build_and_save_reconstruction_dataset(istrain=True, noise=noise)
    build_and_save_reconstruction_dataset(istrain=False, noise=noise)

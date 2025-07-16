import SimpleITK as sitk
import numpy as np
import os
from skimage.transform import radon, iradon
import matplotlib.pyplot as plt

# Set input and output paths
input_file = "./data/patient.nii.gz"
output_dir = "./results"

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read the .nii.gz file
try:
    image = sitk.ReadImage(input_file)
    data = sitk.GetArrayFromImage(image)
except Exception as e:
    print(f"Error reading .nii.gz file: {e}")
    exit(1)

# Check data dimensions
print(f"Data shape: {data.shape}")
num_slices = data.shape[0]

# Parameters for FBP reconstruction
filter_type = "ramp"  # Filter name used by scikit-image
theta = np.linspace(0., 180., max(data.shape[1:]), endpoint=False)

# Perform forward projection and FBP for each slice
for i in range(num_slices):
    try:
        # Extract a single slice
        slice_data = data[i, :, :].astype(np.float32)

        # Forward projection (FP): generate sinogram
        sinogram = radon(slice_data, theta=theta, circle=False)

        # Apply FBP reconstruction
        recon = iradon(sinogram, theta=theta, filter_name=filter_type, output_size=slice_data.shape[1])

        # Normalize the reconstructed image to 0–255
        recon = (recon - np.min(recon)) / (np.max(recon) - np.min(recon)) * 255
        recon = recon.astype(np.uint8)

        # Save as PNG
        output_path = os.path.join(output_dir, f"slice_{i:04d}.png")
        plt.imsave(output_path, recon, cmap='gray')
        print(f"Saved reconstructed slice {i} to {output_path}")
    except Exception as e:
        print(f"Error processing slice {i}: {e}")
        continue

print("Forward projection, reconstruction, and saving completed.")
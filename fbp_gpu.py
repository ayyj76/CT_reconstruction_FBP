import SimpleITK as sitk
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import astra
import os
from scipy.ndimage import gaussian_filter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set input and output paths
input_file = "./data/patient.nii.gz"
output_dir = "./results"
sinogram_dir = "./results/sinograms"

# Ensure output directories exist
for dir_path in [output_dir, sinogram_dir]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Check CUDA availability
try:
    cp.cuda.runtime.getDeviceCount()
    logging.info("CUDA is available.")
except Exception as e:
    logging.error(f"CUDA initialization failed: {e}")
    exit(1)

# Read the .nii.gz file
try:
    image = sitk.ReadImage(input_file)
    data = sitk.GetArrayFromImage(image)
except Exception as e:
    logging.error(f"Error reading .nii.gz file: {e}")
    exit(1)

# Check data dimensions
logging.info(f"Data shape: {data.shape}")
if len(data.shape) != 3:
    logging.error("Expected 3D data with shape (num_slices, height, width).")
    exit(1)
num_slices, height, width = data.shape

# FBP reconstruction parameters
num_projections = 720  # Number of projection angles
theta = np.linspace(0., 180., num_projections, endpoint=False)

# Astra-Toolbox configuration
det_count = width  # Detector count
det_width = 1.0  # Detector pixel size
angles = cp.array(theta * np.pi / 180.0)  # Convert to radians
proj_geom = astra.create_proj_geom('parallel', det_width, det_count, angles.get())
vol_geom = astra.create_vol_geom(height, width)
projector_id = astra.create_projector('cuda', proj_geom, vol_geom)

# Process each slice with FP and FBP
for i in range(num_slices):
    try:
        # Extract and preprocess a single slice
        slice_data = data[i, :, :].astype(np.float32)
        # Normalize input data
        if np.max(slice_data) != np.min(slice_data):
            slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))
        else:
            logging.warning(f"Slice {i} has no dynamic range, skipping normalization.")
        # Apply light Gaussian smoothing for denoising
        slice_data = gaussian_filter(slice_data, sigma=0.5)
        slice_data = cp.array(slice_data)

        # Create Astra data objects
        vol_id = astra.data2d.create('-vol', vol_geom, slice_data.get())
        sino_id = astra.data2d.create('-sino', proj_geom, 0)

        # Forward projection (FP)
        cfg_fp = astra.astra_dict('FP_CUDA')
        cfg_fp['ProjectorId'] = projector_id
        cfg_fp['VolumeDataId'] = vol_id
        cfg_fp['ProjectionDataId'] = sino_id
        alg_id_fp = astra.algorithm.create(cfg_fp)
        astra.algorithm.run(alg_id_fp)
        sinogram = cp.array(astra.data2d.get(sino_id))

        # Save sinogram for debugging
        sinogram_np = sinogram.get()
        if np.max(sinogram_np) != np.min(sinogram_np):
            sinogram_np = (sinogram_np - np.min(sinogram_np)) / (np.max(sinogram_np) - np.min(sinogram_np)) * 255
        sinogram_np = sinogram_np.astype(np.uint8)
        sinogram_path = os.path.join(sinogram_dir, f"sinogram_{i:04d}.png")
        plt.imsave(sinogram_path, sinogram_np, cmap='gray')
        logging.info(f"Saved sinogram {i} to {sinogram_path}")

        # Filtered backprojection (FBP)
        cfg_fbp = astra.astra_dict('FBP_CUDA')
        cfg_fbp['ProjectorId'] = projector_id
        cfg_fbp['ProjectionDataId'] = sino_id
        cfg_fbp['ReconstructionDataId'] = vol_id
        cfg_fbp['option'] = {'FilterType': 'Shepp-Logan'}  # Use smoother filter
        alg_id_fbp = astra.algorithm.create(cfg_fbp)
        astra.algorithm.run(alg_id_fbp)
        recon = cp.array(astra.data2d.get(vol_id))

        # Normalize reconstructed image to 0–255
        recon = recon.get()
        if np.max(recon) != np.min(recon):
            recon = (recon - np.min(recon)) / (np.max(recon) - np.min(recon)) * 255
        recon = recon.astype(np.uint8)

        # Save reconstructed image
        output_path = os.path.join(output_dir, f"slice_{i:04d}.png")
        plt.imsave(output_path, recon, cmap='gray')
        logging.info(f"Saved reconstructed slice {i} to {output_path}")

        # Clean up Astra objects
        astra.algorithm.delete(alg_id_fp)
        astra.algorithm.delete(alg_id_fbp)
        astra.data2d.delete(vol_id)
        astra.data2d.delete(sino_id)

    except Exception as e:
        logging.error(f"Error processing slice {i}: {e}")
        continue

# Clean up projector
astra.projector.delete(projector_id)
logging.info("CUDA-accelerated forward projection, reconstruction, and saving completed.")
#dhcp_21
import os
import nibabel as nib
import numpy as np
import glob

# Define input and output folders
input_folder = "/opt/localdata/data/M_CRIB/whole_brain/parc"  # Change this to the actual root folder
output_folder = "/opt/localdata/data/usr-envs/MAPSEG/MAPSeg-main/tissue_data/MCRIB_label"  # Change this to the output folder (flat structure)

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Find all _seg.nii files in subfolders
nii_files = glob.glob(os.path.join(input_folder, "**", "*.nii"), recursive=True)
# Define the mapping of values
# #feta
# value_mapping = {
#     (1):1,
#     (2):2,
#     (3):3
    
# }

# #dhcp_2136
# value_mapping = {
#     (1,2):1,
#     (3,4):2,
#     (5,6):3
    
# }
#dhcp
value_mapping = {
    (24):1,
    
    (2,41):3
    
}
value_mapping.update({i: 2 for i in range(1000, 2036)})  
# value_mapping = {
#     (24): 1,
#     # (2, 41): 2,
#     (2, 41): 2,
#     (4, 43): 3,
#     (9, 48): 4,
#     (170,): 5
# }

for file_path in nii_files:
    # Load the NIfTI file
    nii = nib.load(file_path)

    # Get the image data as a NumPy array
    data = nii.get_fdata()

    # Apply the value mapping
    modified_data = np.zeros_like(data)  # Initialize all values to 0
    for old_values, new_value in value_mapping.items():
        modified_data[np.isin(data, old_values)] = new_value

    # Extract filename and remove `_seg`
    # filename = os.path.basename(file_path).replace("_2-0", "").replace("_parc","_T1") + ".gz"
    filename = os.path.basename(file_path).replace("2-0", "").replace("parc","T1")+".gz"
    new_file_path = os.path.join(output_folder, filename)

    # Save the modified NIfTI file in the flat output folder
    new_nii = nib.Nifti1Image(modified_data, affine=nii.affine, header=nii.header)
    nib.save(new_nii, new_file_path)

    print(f"Processed and saved: {file_path} → {new_file_path}")

print("All _seg.nii files processed and saved successfully!")

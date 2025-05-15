import pandas as pd

# Define the data
data = {
    'Filename': ['sub-HCA6002236_ses-01_T2w_brain.nii.gz'],
    'Age': [10],
    'Sex': [1]  # 1 = Male, 0 = Female (assumed convention)
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to Excel
df.to_excel("../test_dataset.xls", index=False)


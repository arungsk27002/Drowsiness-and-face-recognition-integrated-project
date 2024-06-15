import os

# Specify the directory path
directory = r"C:\Users\HP\OneDrive\Documents\dixon\ai\images"

# Get all the files in the directory
files = os.listdir(directory)

# Sort the files alphabetically
files.sort()

# Rename the files starting with "1"
for i, file in enumerate(files):
    # Get the current file extension
    file_extension = os.path.splitext(file)[1]
    
    # Generate the new file name
    new_file_name = f'1{i+1}{file_extension}'
    
    # Rename the file
    os.rename(os.path.join(directory, file), os.path.join(directory, new_file_name))

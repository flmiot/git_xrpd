import os
import re
import numpy as np
import tifffile as tiff

def read_images(image_file_path, run_nr):
    files = sorted(os.listdir(image_file_path))
    pattern = r'run04_20_(\d{5})'

    image_count = 0
    filenames = []
    for file in files:
        if re.findall(pattern, file )[0] == run_nr:
            image_count += 1
            filenames.append(file)



    images = np.empty((image_count, 195, 487))

    for idx, file in enumerate(filenames):
        path = os.path.join(image_file_path, file)
        images[idx] = tiff.imread(path)

    return images

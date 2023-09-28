import shutil
import os

# Clear all data in the DATA folder

ROOT = '/home/timssh/ML/TAGGING/DATA'
PATHS = ['segmentation/boxes', 'segmentation/picture', 'masks', 'meta', 'picture']

for path in PATHS:
    shutil.rmtree(os.path.join(ROOT, path))
    os.makedirs(os.path.join(ROOT, path))

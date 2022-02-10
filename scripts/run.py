import os

# Prepare dataset
os.system("python ./prepare_dataset.py --images_dir ../data/T91/original --output_dir ../data/T91/ESPCN/train --image_size 80 --step 40 --num_workers 10")

# Split train and valid
os.system("python ./split_train_valid_dataset.py --train_images_dir ../data/T91/ESPCN/train --valid_images_dir ../data/T91/ESPCN/valid --valid_samples_ratio 0.1")

# Create LMDB database file
os.system("python ./create_lmdb_dataset.py --images_dir ../data/T91/ESPCN/train --lmdb_path ../data/train_lmdb/ESPCN/T91_HR_lmdb --upscale_factor 1")
os.system("python ./create_lmdb_dataset.py --images_dir ../data/T91/ESPCN/train --lmdb_path ../data/train_lmdb/ESPCN/T91_LRbicx2_lmdb --upscale_factor 2")
os.system("python ./create_lmdb_dataset.py --images_dir ../data/T91/ESPCN/train --lmdb_path ../data/train_lmdb/ESPCN/T91_LRbicx3_lmdb --upscale_factor 3")
os.system("python ./create_lmdb_dataset.py --images_dir ../data/T91/ESPCN/train --lmdb_path ../data/train_lmdb/ESPCN/T91_LRbicx4_lmdb --upscale_factor 4")

os.system("python ./create_lmdb_dataset.py --images_dir ../data/T91/ESPCN/valid --lmdb_path ../data/valid_lmdb/ESPCN/T91_HR_lmdb --upscale_factor 1")
os.system("python ./create_lmdb_dataset.py --images_dir ../data/T91/ESPCN/valid --lmdb_path ../data/valid_lmdb/ESPCN/T91_LRbicx2_lmdb --upscale_factor 2")
os.system("python ./create_lmdb_dataset.py --images_dir ../data/T91/ESPCN/valid --lmdb_path ../data/valid_lmdb/ESPCN/T91_LRbicx3_lmdb --upscale_factor 3")
os.system("python ./create_lmdb_dataset.py --images_dir ../data/T91/ESPCN/valid --lmdb_path ../data/valid_lmdb/ESPCN/T91_LRbicx4_lmdb --upscale_factor 4")

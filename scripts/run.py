import os

# Prepare dataset
os.system("python3 ./prepare_dataset.py --inputs_dir ../data/T91/original --output_dir ../data/T91/ESPCN --upscale_factor 2")
os.system("python3 ./prepare_dataset.py --inputs_dir ../data/T91/original --output_dir ../data/T91/ESPCN --upscale_factor 3")
os.system("python3 ./prepare_dataset.py --inputs_dir ../data/T91/original --output_dir ../data/T91/ESPCN --upscale_factor 4")

# Split train and valid
os.system("python3 ./split_train_valid_dataset.py --images_dir ../data/T91/ESPCN --upscale_factor 2")
os.system("python3 ./split_train_valid_dataset.py --images_dir ../data/T91/ESPCN --upscale_factor 3")
os.system("python3 ./split_train_valid_dataset.py --images_dir ../data/T91/ESPCN --upscale_factor 4")

# Create LMDB database file
os.system("python3 ./create_lmdb_database.py --images_dir ../data/T91/ESPCN/x2/train --lmdb_path ../data/train_lmdb/ESPCN/x2/T91_HR_lmdb --upscale_factor 1")
os.system("python3 ./create_lmdb_database.py --images_dir ../data/T91/ESPCN/x3/train --lmdb_path ../data/train_lmdb/ESPCN/x3/T91_HR_lmdb --upscale_factor 1")
os.system("python3 ./create_lmdb_database.py --images_dir ../data/T91/ESPCN/x4/train --lmdb_path ../data/train_lmdb/ESPCN/x4/T91_HR_lmdb --upscale_factor 1")
os.system("python3 ./create_lmdb_database.py --images_dir ../data/T91/ESPCN/x2/train --lmdb_path ../data/train_lmdb/ESPCN/x2/T91_LR_lmdb --upscale_factor 2")
os.system("python3 ./create_lmdb_database.py --images_dir ../data/T91/ESPCN/x3/train --lmdb_path ../data/train_lmdb/ESPCN/x3/T91_LR_lmdb --upscale_factor 2")
os.system("python3 ./create_lmdb_database.py --images_dir ../data/T91/ESPCN/x4/train --lmdb_path ../data/train_lmdb/ESPCN/x4/T91_LR_lmdb --upscale_factor 2")
os.system("python3 ./create_lmdb_database.py --images_dir ../data/T91/ESPCN/x2/train --lmdb_path ../data/train_lmdb/ESPCN/x2/T91_LR_lmdb --upscale_factor 3")
os.system("python3 ./create_lmdb_database.py --images_dir ../data/T91/ESPCN/x3/train --lmdb_path ../data/train_lmdb/ESPCN/x3/T91_LR_lmdb --upscale_factor 3")
os.system("python3 ./create_lmdb_database.py --images_dir ../data/T91/ESPCN/x4/train --lmdb_path ../data/train_lmdb/ESPCN/x4/T91_LR_lmdb --upscale_factor 3")
os.system("python3 ./create_lmdb_database.py --images_dir ../data/T91/ESPCN/x2/train --lmdb_path ../data/train_lmdb/ESPCN/x2/T91_LR_lmdb --upscale_factor 4")
os.system("python3 ./create_lmdb_database.py --images_dir ../data/T91/ESPCN/x3/train --lmdb_path ../data/train_lmdb/ESPCN/x3/T91_LR_lmdb --upscale_factor 4")
os.system("python3 ./create_lmdb_database.py --images_dir ../data/T91/ESPCN/x4/train --lmdb_path ../data/train_lmdb/ESPCN/x4/T91_LR_lmdb --upscale_factor 4")

os.system("python3 ./create_lmdb_database.py --images_dir ../data/T91/ESPCN/x2/valid --lmdb_path ../data/valid_lmdb/ESPCN/x2/T91_HR_lmdb --upscale_factor 1")
os.system("python3 ./create_lmdb_database.py --images_dir ../data/T91/ESPCN/x3/valid --lmdb_path ../data/valid_lmdb/ESPCN/x3/T91_HR_lmdb --upscale_factor 1")
os.system("python3 ./create_lmdb_database.py --images_dir ../data/T91/ESPCN/x4/valid --lmdb_path ../data/valid_lmdb/ESPCN/x4/T91_HR_lmdb --upscale_factor 1")
os.system("python3 ./create_lmdb_database.py --images_dir ../data/T91/ESPCN/x2/valid --lmdb_path ../data/valid_lmdb/ESPCN/x2/T91_LR_lmdb --upscale_factor 2")
os.system("python3 ./create_lmdb_database.py --images_dir ../data/T91/ESPCN/x3/valid --lmdb_path ../data/valid_lmdb/ESPCN/x3/T91_LR_lmdb --upscale_factor 2")
os.system("python3 ./create_lmdb_database.py --images_dir ../data/T91/ESPCN/x4/valid --lmdb_path ../data/valid_lmdb/ESPCN/x4/T91_LR_lmdb --upscale_factor 2")
os.system("python3 ./create_lmdb_database.py --images_dir ../data/T91/ESPCN/x2/valid --lmdb_path ../data/valid_lmdb/ESPCN/x2/T91_LR_lmdb --upscale_factor 3")
os.system("python3 ./create_lmdb_database.py --images_dir ../data/T91/ESPCN/x3/valid --lmdb_path ../data/valid_lmdb/ESPCN/x3/T91_LR_lmdb --upscale_factor 3")
os.system("python3 ./create_lmdb_database.py --images_dir ../data/T91/ESPCN/x4/valid --lmdb_path ../data/valid_lmdb/ESPCN/x4/T91_LR_lmdb --upscale_factor 3")
os.system("python3 ./create_lmdb_database.py --images_dir ../data/T91/ESPCN/x2/valid --lmdb_path ../data/valid_lmdb/ESPCN/x2/T91_LR_lmdb --upscale_factor 4")
os.system("python3 ./create_lmdb_database.py --images_dir ../data/T91/ESPCN/x3/valid --lmdb_path ../data/valid_lmdb/ESPCN/x3/T91_LR_lmdb --upscale_factor 4")
os.system("python3 ./create_lmdb_database.py --images_dir ../data/T91/ESPCN/x4/valid --lmdb_path ../data/valid_lmdb/ESPCN/x4/T91_LR_lmdb --upscale_factor 4")

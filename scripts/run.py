import os

os.system("python .\create_lmdb_database.py --image_dir T91/x2/train --lmdb_path train_lmdb/ESPCN/x2/T91_HR_lmdb --upscale_factor 1")
os.system("python .\create_lmdb_database.py --image_dir T91/x3/train --lmdb_path train_lmdb/ESPCN/x3/T91_HR_lmdb --upscale_factor 1")
os.system("python .\create_lmdb_database.py --image_dir T91/x4/train --lmdb_path train_lmdb/ESPCN/x4/T91_HR_lmdb --upscale_factor 1")

os.system("python .\create_lmdb_database.py --image_dir T91/x2/valid --lmdb_path valid_lmdb/ESPCN/x2/T91_HR_lmdb --upscale_factor 1")
os.system("python .\create_lmdb_database.py --image_dir T91/x3/valid --lmdb_path valid_lmdb/ESPCN/x3/T91_HR_lmdb --upscale_factor 1")
os.system("python .\create_lmdb_database.py --image_dir T91/x4/valid --lmdb_path valid_lmdb/ESPCN/x4/T91_HR_lmdb --upscale_factor 1")

os.system("python .\create_lmdb_database.py --image_dir T91/x2/train --lmdb_path train_lmdb/ESPCN/x2/T91_LR_lmdb --upscale_factor 2")
os.system("python .\create_lmdb_database.py --image_dir T91/x3/train --lmdb_path train_lmdb/ESPCN/x3/T91_LR_lmdb --upscale_factor 2")
os.system("python .\create_lmdb_database.py --image_dir T91/x4/train --lmdb_path train_lmdb/ESPCN/x4/T91_LR_lmdb --upscale_factor 2")

os.system("python .\create_lmdb_database.py --image_dir T91/x2/valid --lmdb_path valid_lmdb/ESPCN/x2/T91_LR_lmdb --upscale_factor 2")
os.system("python .\create_lmdb_database.py --image_dir T91/x3/valid --lmdb_path valid_lmdb/ESPCN/x3/T91_LR_lmdb --upscale_factor 2")
os.system("python .\create_lmdb_database.py --image_dir T91/x4/valid --lmdb_path valid_lmdb/ESPCN/x4/T91_LR_lmdb --upscale_factor 2")

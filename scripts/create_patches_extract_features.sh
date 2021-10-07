#Fast pipeline

PATCH_SIZE=1024

#python create_patches_fp.py --source /mnt/storage/COMET/RAW --save_dir /mnt/storage/COMET/preprocessed_test1024_fp --patch_size 1024 --step_size 1024 --seg --patch --stitch 
python create_patches_fp.py --source /mnt/storage/COMET/RAW --save_dir /mnt/storage/COMET/preprocessed_test1024_stepsize_512_fp --patch_size 1024 --step_size 512 --seg --patch --stitch 
#python create_patches_fp.py --source /mnt/storage/COMET/RAW --save_dir /mnt/storage/COMET/preprocessed_test_fp --patch_size 256 --step_size 256 --seg --patch --stitch 

#Slow pipeline
#python create_patches.py --source /mnt/storage/COMET/RAW --save_dir /mnt/storage/COMET/preprocessed_test --patch_size 256 --seg --patch --stitch 

CUDA_VISIBLE_DEVICES=1 python extract_features_fp.py --data_h5_dir /home/blansdell/COMET/preprocessed_test1024_fp --data_slide_dir /home/blansdell/COMET/RAW --csv_path cometFiltered.csv --feat_dir /home/blansdell/COMET/preprocessed_test1024_fp/features --batch_size 8 --slide_ext .svs
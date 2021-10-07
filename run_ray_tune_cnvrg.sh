#Final installs
pip3 install torch torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html 
pip install ray[default]
pip install ray[tune]

#CNVRG test run
# python main_ray_tune_cnvrg.py \
#  --drop_out --lr 1e-4 --reg 1e-4 --k 10 --max_epochs 10 --label_frac 1 --k_start 0 --k_end 1 --early_stopping \
#  --exp_code task_2_tumor_subtyping_1024_lr1e-4_reg1e-4_adamw_CLAM_100 --weighted_sample --bag_loss ce \
#  --inst_loss svm --task task_2_tumor_subtyping \
#  --model_type clam_sb --log_data --subtyping \
#  --save_activations \
#  --split_dir /cnvrg/CLAM/splits/task_2_tumor_subtyping_100/ \
#  --data_root_dir /data/comet_rms/preprocessed_test1024_fp/features/

#cnvrg main run
python main_ray_tune_cnvrg.py \
 --drop_out --lr 1e-4 --reg 1e-4 --k 10 --max_epochs 100 --label_frac 1 --k_start 0 --k_end 10 --early_stopping \
 --exp_code task_2_tumor_subtyping_1024_lr1e-4_reg1e-4_adamw_CLAM_100 --weighted_sample --bag_loss ce \
 --inst_loss svm --task task_2_tumor_subtyping --split_dir /cnvrg/CLAM/splits/task_2_tumor_subtyping_100 \
 --model_type clam_sb --log_data --subtyping --data_root_dir /data/comet_rms/preprocessed_1024_fp_allbatches/features/
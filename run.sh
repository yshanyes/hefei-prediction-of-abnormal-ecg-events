
python round1_data_process.py

python round1_beat_process.py

python round2_data_process.py

python round2_beat_process.py

python main.py train --model_name=mixnet_sm_pretrain

python main.py train_cv --model_name=mixnet_sm

python main.py train --model_name=mixnet_mm_pretrain

python main.py train_cv --model_name=mixnet_mm

python main.py val_cv --ckpt=ckpt/mixnet_sm_cv --model_name=mixnet_sm_predict

python main.py val_cv --ckpt=ckpt/mixnet_mm_cv --model_name=mixnet_mm_predict

python main.py test_cv_ensemble --ckpt=ckpt




#python main.py train --model_name=mixnet_sm_pretrain

#python main.py train_cv --model_name=mixnet_sm

#python main.py testxbeat --ckpt=./ckpt/resnet34_201911081635/best_weight.pth
#python main.py test_cv --ckpt=./ckpt/mixnet_mm_cv_201911110948
#python main.py test_cv --ckpt=./ckpt/mixnet_sm_cv
#python main.py test_cv_ensemble --ckpt=./ckpt
#python main.py test_ensemble --ckpt=./ckpt

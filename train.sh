python train.py --dataroot ./datasets/opt2sar --name opt2sar_1015 --model cycle_gan --preprocess scale_width_and_crop --load_size 800 --crop_size 360

python test.py --dataroot /mnt/hdd1/xueyijun/Vicy/20250612/data/20250924/ChangGuang/fleet/images --name opt2sar_new --model test --no_dropout --preprocess scale_width --load_size 800

python test.py --dataroot /app/dataroot/output/fleet/images --name opt2sar --model test --no_dropout --preprocess scale_width --load_size 2048 --num_test 1000
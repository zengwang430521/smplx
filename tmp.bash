conda de
source /mnt/lustre/share/polaris/env/pt1.5s1
srun -p HA_3D --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=data_process --kill-on-bad-exit=1 python debug.py --exp-cfg ../config_files/smpl2smplx.yaml --exp-opts batch_size=256  --begin-index=0
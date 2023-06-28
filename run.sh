#cd /homes/80/kang/SpeedTesting/
#touch OutPut4.1.txt
#touch OutPut4.2.txt

python -m cProfile -s cumulative /homes/80/kang/SpeedTesting/train/sb3/train_v4.1.py > /homes/80/kang/SpeedTesting/OutPut4.1.txt
python -m cProfile -s cumulative /homes/80/kang/SpeedTesting/train/sb3/train_v4.2.py > /homes/80/kang/SpeedTesting/OutPut4.2.txt


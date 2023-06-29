#cd /homes/80/kang/SpeedTesting/
#touch OutPut4.1.txt
#touch OutPut4.2.txt

python -m cProfile -s cumulative /homes/80/kang/SpeedTesting/train/sb3/train_v4.1.py > /homes/80/kang/SpeedTesting/OutPut4.1.txt
python -m cProfile -s cumulative /homes/80/kang/SpeedTesting/train/sb3/train_v4.2.py > /homes/80/kang/SpeedTesting/OutPut4.2.txt


python -m cProfile -s cumulative /homes/80/kang/SpeedTesting/train/sb3/train_v4.1.1.py > /homes/80/kang/SpeedTesting/OutPut4.1.1.txt

#/homes/80/kang/SpeedTesting/train/sb3/train_v4.1.1.py

#cd /homes/80/kang/SpeedTesting/
#touch OutPut4.1.1.txt
#touch OutPut4.1.2.txt


python -m cProfile -s cumulative /homes/80/kang/SpeedTesting/train/sb3/train_v4.1.1.py > /homes/80/kang/SpeedTesting/OutPut4.1.1.txt

python -m cProfile -s cumulative /homes/80/kang/SpeedTesting/train/sb3/train_v4.1.2.py > /homes/80/kang/SpeedTesting/OutPut4.1.2.txt



#cd /home/kanli/speedTesting/
#touch OutPut4.1.2.2.txt
python -m cProfile -s cumulative /home/kanli/speedTesting/train/sb3/train_v4.1.2.2.py > /home/kanli/speedTesting/OutPut4.1.2.2.txt

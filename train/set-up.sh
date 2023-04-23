cd ~/AlphaTrade/
git pull

export PYTHONPATH=$PYTHONPATH:/home/duser/AlphaTrade
pip install sb3-contrib
pip install tensorboard
pip install sortedcontainers
pip install wandb
wandb login 41f4ee88a220359a48d63a1a4239c83862288bb0

cd ~/AlphaTrade/
mkdir data
docker cp AMZN_2021-04-01_34200000_57600000_orderbook_10.csv *******:/home/duser/AlphaTrade/data/
docker cp AMZN_2021-04-01_34200000_57600000_message_10.csv ********:/home/duser/AlphaTrade/data/


cd ~/AlphaTrade/train
python3 train_v1.py



cd ~/AlphaTrade/
git pull
export PYTHONPATH=$PYTHONPATH:/home/duser/AlphaTrade
python3 ./train/train_v1.py

import sys
sys.path.append('/Users/sasrey/AlphaTrade')
import gymnax_exchange
import gym_exchange


from gym_exchange.data_orderbook_adapter.raw_encoder import RawDecoder, RawEncoder
from gym_exchange.data_orderbook_adapter.decoder import Decoder
from gym_exchange.data_orderbook_adapter.encoder import Encoder
from gym_exchange.data_orderbook_adapter.data_pipeline import DataPipeline

print('imports complete')



print(DataPipeline().data_loader)

decoder = RawDecoder(**DataPipeline()())
encoder = RawEncoder(decoder)


print(encoder())
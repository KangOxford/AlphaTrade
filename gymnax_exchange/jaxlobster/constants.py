from dataclasses import dataclass
import numpy as np

TIME_COL = "<time>"
EVENT_TYPE_COL = "<event_type>"
ORDER_ID_COL = "<order_id>"
SIZE_COL = "<size>"
PRICE_COL = "<price>"
DIRECTION_COL = "<direction>"

MESSAGE_TOKEN_DTYPE_MAP = {
    TIME_COL: int,  
    EVENT_TYPE_COL: int,
    ORDER_ID_COL: int,
    SIZE_COL: int,
    PRICE_COL: int,
    DIRECTION_COL: int
}
MESSAGE_TOKEN_TYPES = list(MESSAGE_TOKEN_DTYPE_MAP.keys())

def get_orderbook_token_types(levels: int) -> list[str]: 
    return np.array([
                [f"<ask_price_{i}>", f"<ask_size_{i}>", f"<bid_price_{i}>", f"<bid_size_{i}>"]
                for i in range(1, levels + 1)]
            ).flatten().tolist()

@dataclass
class MambaTrainArgs:
    train_data_dir: str = "./data/raw/"
    eval_data_dir: str = "./data/test/"
    file_filter_train: str = ""
    file_filter_eval: str = ""
    save_path: str = "./models/mamba2"
    nmsgs: int = 50
    only_use_message_orderbook_matches: bool = True

    tokenizer_file: str = "tokenizers/lob_tok_with_time_diff.json"

    wandb_online: bool = True
    wandb_project: str = "lobgen"
    wandb_entity: str = "gereon-franken-oxford"



@dataclass
class MambaInferenceArgs:
    model_path: str
    tokenizer_path: str = "tokenizers/lob_tok_messages.json"
    is_sharded: bool = False
    test_dir: str = "data/GOOG/test/"
    test_filter: str = "2018-12-31"
    genlen: int = 100
    iterations: int = 10
    temperature: float = 1.0
    topk: int = 50
    topp: float = 1.0
    minp: float = 0.0
    repetition_penalty: float = 1.0
    batch: int = 1

@dataclass
class MambaBenchmarkingArgs(MambaInferenceArgs):
    data_dir: str = "data/GOOG/2018/"
    data_time_stamp: str = "2018-12-31"
    save_path: str = "gen_data/"

@dataclass
class TokenizerTrainArgs:
    data_dir: str = "./data/raw/"
    file_filter: str = "*.csv"
    save_path: str = "./tokenizers/lob_tok.json"
    vocab_size: int = 10_000
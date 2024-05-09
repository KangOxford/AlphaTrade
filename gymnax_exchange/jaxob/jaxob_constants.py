
from enum import Enum

class MaxInt(Enum):
    _64_Bit_Signed=2_147_483_647
    _32_Bit_Signed=32_767

INITID=-900000

class CancelMode(Enum):
    STRICT_BY_ID=0 #Cancel only if ID matches. 
    INCLUDE_INITS=1 #Cancel only init orders if ID does not match
    CANCEL_UNIFORM=2  #Pick a random order at the right price level to cancel 
    CANCEL_WEIGHTED=3 # Unused for now

SEED= 42 # the meaning of life. 

import gymnax_exchange.jaxob.jaxob_constants as cst
import jax

class Configuration:
    
    def __init__(self,):
        self.maxint= cst.MaxInt._64_Bit_Signed
        self.init_id = cst.INITID
        self.cancel_mode= cst.CancelMode.CANCEL_UNIFORM
        self.mainkey=jax.random.PRNGKey(cst.SEED)
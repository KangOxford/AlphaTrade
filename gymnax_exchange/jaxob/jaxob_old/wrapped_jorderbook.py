import importlib
import gymnax_exchange.jaxob.JaxOrderbook as job
job=importlib.reload(job)
from gymnax_exchange.jaxob.jorderbook import OrderBook as OB


class OrderBook(OB):
    def __init__(self):
        super().__init__()

    @property
    def asks(self):
        return self.orderbook_array[0]

    @property
    def bids(self):
        return self.orderbook_array[1]

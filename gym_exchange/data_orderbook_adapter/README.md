## Outlines for the code:

1 `data_pipeline.py` used for defining the data path of your reading data. You should config it before runnning.

2 `decoder.py` is used for transferring the data into the trading signals 
  * `decoder = Decoder(**DataPipeline()())` here `DataPipeline()` genereate an instance. `DataPipeline()()` call the `__call__()` method. `**DataPipeline()()` unpack the return tuple of the `__call__()` method of class DataPipeline.
  * `signals_list = decoder.process()` the `process()` is the main method which would be used in the class Decoder. Here `decoder` is an instance of `Decoder`.
  * the `trading signals` is produce by two parts: `InsideSignalEncoder` and `OutsideSignalEncoder`
    * the `InsideSignal` means the trading signals inside the price level (e.g. 10 for our example data)
    * the `OutsideSignal` means the trading signals outside the price level (e.g. 10 for our example data)
  * class `DataAdjuster` is another important part of the decoder, which helps `adjust_data_drift`. It means that the l2 and l3 data may contains different info, and this class is usd for extracting the useful info which might be in the l2 data, but not l3.
    * e.g. some order is partly cancelled outside the price level
  * `SignalPorcessor` is a class used to ensure the trading signals is right by executing the trading signals and then get the orderbook and then compare the generated orderbook with the l2 data.
  
3 `encoder.py` is used for transfer the `trading signals` into `order flows`, the later is the machine readable for the package `orderbook/jaxob`
  * `decoder = Decoder(**DataPipeline()()); encoder = Encoder(decoder); Ofs = encoder()`

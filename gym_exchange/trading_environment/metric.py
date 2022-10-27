class Vwap():
    time_window = 1 # one step
    def __init__(self, historical_data, running_data):
        self.historical_data = historical_data
        self.running_data = running_data
        
    @property
    def difference(self):
        pass
    
    class StaticVwap():
        def __init__(self, historical_data, running_data):
            pass
    
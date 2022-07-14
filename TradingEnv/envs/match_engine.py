# %%
import pandas as pd
# import numpy as np
Dir = "/workspaces/Dissertation/TradingEnv/envs/"
filename = "AMZN_2021-04-01_34200000_57600000_orderbook_10.csv"
name_lst = []
for i in range(40//4):
    name_lst.append("ask"+str(i+1))
    name_lst.append("ask"+str(i+1)+"_quantity")
    name_lst.append("bid"+str(i+1))
    name_lst.append("bid"+str(i+1)+"_quantity")
    
Flow = pd.read_csv(Dir + filename, names= name_lst)
column_numbers=[i for i in range(40) if i%4==2 or i%4==3]
flow = Flow.iloc[:,column_numbers]
# %%
-flow.diff()
# %%
# if __name__ == "__main__":
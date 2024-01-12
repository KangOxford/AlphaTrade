import sys
from gymnax_exchange.jaxrl.VWAP_Scheduling import data_alignment
from gymnax_exchange.jaxen.base_env import load_LOBSTER
import pickle

def process_data(symbol):
    ATFolder = f"/homes/80/kang/SP500/{symbol}_data"
    common_dates, common_stocks, VWAPs, ORACLEs, RMs, TWAPs = data_alignment(ATFolder)
    
    A = {
        "sliceTimeWindow": 1800,
        'stepLines': 100, 
        'messagePath': ATFolder+"/Flow_10/",
        'orderbookPath': ATFolder+"/Flow_10/",
        'start_time': 34200,
        'end_time': 57600,
        'dates': common_dates,
    }

    Cubes_withOB, max_steps_in_episode_arr, start_idx_array_list, taskSize_array = load_LOBSTER(
        A['sliceTimeWindow'],
        A['stepLines'],
        A['messagePath'],
        A['orderbookPath'],
        A['start_time'],
        A['end_time'],
        A['dates']
    )
    
    # Save to a pickle file
    with open(f'saved_objects_{symbol}.pkl', 'wb') as f:
        pickle.dump((Cubes_withOB, max_steps_in_episode_arr, start_idx_array_list, taskSize_array), f)

# Check if a command-line argument is provided
if len(sys.argv) > 1:
    symbol = sys.argv[1]
    process_data(symbol)
else:
    print("Please provide a stock symbol as a command-line argument.")

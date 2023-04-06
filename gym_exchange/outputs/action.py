input="/Users/kang/AlphaTrade/gym_exchange/outputs/actions"
with open(input, 'r') as file:
    text = file.readlines()
import re
import numpy as np
pattern = r"\[[-\s\d]+\]"
result = re.findall(pattern, "".join(text))
arr = np.array([np.fromstring(x.strip('[]'), sep=' ', dtype=np.int64) for x in result])
with open(input, 'w+') as file:
    for row in arr:
        file.write(' '.join([str(elem) for elem in row]))
        file.write('\n')

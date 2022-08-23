#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 13:06:06 2022

@author: kang
"""
# %%
import pandas as pd
import numpy as np

df = pd.read_csv("/Users/kang/Downloads/analysis_test_cell_gym-v18.txt",sep = " ", header = None)
df1 = df.drop(columns=df.columns[[0, 2, 4]])
df2 = df1.set_axis(['is_overlap', 'overlap_potential', 'packing_fraction'],axis=1)
df3 = df2[df2.overlap_potential <=1e-2]
df4 = df3.sort_values(by = ['packing_fraction'], ascending=False)
df4.to_csv("/Users/kang/Downloads/analysis_test_cell_gym-v18.csv")


# %%
import pandas as pd
import numpy as np

df = pd.read_csv("/Users/kang/Downloads/analysis_test_cell_gym-v18.3.txt",sep = " ", header = None)
df1 = df.drop(columns=df.columns[[0, 2, 4]])
df2 = df1.set_axis(['is_overlap', 'overlap_potential', 'packing_fraction'],axis=1)
df3 = df2[df2.overlap_potential == 0]
df4 = df3.sort_values(by = ['packing_fraction'], ascending=False)
df4.to_csv("/Users/kang/Downloads/analysis_test_cell_gym-v18.3.csv")



# %%
import pandas as pd
import numpy as np

df = pd.read_csv("/Users/kang/Downloads/analysis_test_cell_gym-v18.3.txt",sep = " ", header = None)
df1 = df.drop(columns=df.columns[[0, 2, 4]])
df2 = df1.set_axis(['is_overlap', 'overlap_potential', 'packing_fraction'],axis=1)
df3 = df2[(df2.packing_fraction <=0.7405) & (df2.overlap_potential <= 0.01) ]
df4 = df3.sort_values(by = ['packing_fraction'], ascending=False)
df4.to_csv("/Users/kang/Downloads/analysis_test_cell_gym-v18.3.0csv")

# %%
info_list = []
info_list.append(info)
df = pd.DataFrame(info_list)

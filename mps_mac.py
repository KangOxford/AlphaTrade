#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 10:36:26 2022

@author: kang
"""

import torch
torch.__version__
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
x = torch.tensor([1,2,3])
x = x.to(device)

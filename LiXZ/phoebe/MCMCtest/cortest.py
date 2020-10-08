# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 16:56:04 2020

@author: dingxu
"""

import phoebe
from phoebe import u # units
import numpy as np
import matplotlib.pyplot as plt

logger = phoebe.logger()

b = phoebe.default_binary(contact_binary=True)

print(b.filter(qualifier=['q_max', 'q_min']))

#print(b.filter(qualifier=['requiv_max', 'requiv_min', 'pot_max', 'pot_min', 'q_max', 'q_min']))


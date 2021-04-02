# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 18:11:34 2021

@author: dingxu
"""

import phoebe
from phoebe import u,c

logger = phoebe.logger(clevel='WARNING')
b = phoebe.default_binary(contact_binary=True)


#print(b.filter(context='constraint').qualifiers)

#print(b.get_parameter('mass', component='primary', context='component'))
#print(b.get_parameter('fillout_factor', context='component'))


#b.flip_constraint(qualifier='requiv', context='primary',solve_for='requiv@primary') #qualifier='fillout_factor', component='contact_envelope',
b.flip_constraint(qualifier='fillout_factor', component='contact_envelope', solve_for='q@binary')


#b.flip_constraint(qualifier='fillout_factor', component='contact_envelope', solve_for='q@binary')
print(b.get_constraint(qualifier='fillout_factor', component='contact_envelope'))

#print(b.get_constraint(qualifier='mass', component='primary'))

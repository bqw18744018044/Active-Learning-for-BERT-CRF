# encoding: utf-8
"""
@author: bqw
@time: 2020/5/13 15:39
@file: enums.py
@desc: 
"""
import enum

# query strategy in active learning
STRATEGY = enum.Enum('STRATEGY', ('RAND', 'LC', 'MNLP', 'TTE', 'TE'))
import _pickle as cPickle
# print(overcooked_ai_py.__file__)
# /Users/michellezhao/Documents/replicate_neurips2019_for_saving/human_aware_rl/overcooked_ai/overcooked_ai_py/__init__.py
# hh_all_2019_file = '../human_aware_rl/static/human_data/cleaned/2019_hh_trials_all.pickle'
# # hh_all_2019_file = 'human_aware_rl/static/human_data/cleaned/2019_hh_trials_all.pickle'
# print("hh_all_2019_file", hh_all_2019_file)
#
# with open(hh_all_2019_file, 'rb') as file:
#     humans_2019_file = cPickle.load(file)

import pandas as pd

humans_2019_file = pd.read_pickle(r'../human_aware_rl/static/human_data/cleaned/2019_hh_trials_all.pickle')

print(humans_2019_file)
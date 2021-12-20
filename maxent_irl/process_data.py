from dependencies import *



def json_eval(s):
    json_acceptable_string = s.replace("'", "\"")
    d = json.loads(json_acceptable_string)
    return d



def import_2019_data():
    hh_all_2019_file = '../human_aware_rl/static/human_data/cleaned/2019_hh_trials_all.pickle'

    with open(hh_all_2019_file,'rb') as file:
        humans_2019_file = pkl.load(file)

    # humans_2019_file.to_csv('humans_all_2019.csv')
    old_trials = humans_2019_file
    return old_trials

def import_2020_data():
    hh_all_2020_file = '../human_aware_rl/static/human_data/cleaned/2020_hh_trials_all.pickle'

    with open(hh_all_2020_file,'rb') as file:
        humans_2020_file = pkl.load(file)

    # humans_2020_file.to_csv('humans_all_2020.csv')
    new_trials = humans_2020_file
    return new_trials









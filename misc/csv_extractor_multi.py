import csv
import numpy as np
from lazy_streams import stream

path = '/Users/taha/PycharmProjects/TennesseeEastmanProcess/misc/wandb_export_2021-02-24T14_16_06.088-06_00.csv'

groups = {'do_baseline': None,
          'do_observation_only': 'env_params.compromise_observation_prob',
          'do_memory': 'env_params.history_length',
          'do_power': 'env_params.power',
          'do_start_set': 'env_params.test_env',
          'do_env_noise': 'env_params.noise_sigma',
          'do_actuation_only': 'env_params.compromise_actuation_prob',
          'do_baseline_random_start': None
          }

print(groups.keys())

env = 'BRP'

with open(path) as fd:
    reader = csv.DictReader(fd)
    rows = list(reader)

variables = dict()
for row in rows:
    g = row['Group']
    v = groups[g]
    if v is None:
        continue
    if v in variables:
        variables[v].add(row[v])
    else:
        variables[v] = {row[v]}

improvements = dict()

for g in groups:
    for v in variables[groups[g]] if groups[g] is not None else [None]:
        key = (g, v)

        for row in rows:
            if row['Group'] == g:
                if row['env_id'] == env:
                    if groups[g] is None or row[groups[g]] == v:

                        msne_br = float(row['final_payoff/msne_br'])
                        base_br = float(row['final_payoff/base_br'])
                        no_def = float(row['final_payoff/no_defense'])
                        msne_table = float(row['final_payoff/msne_table'])
                        msne_eval = float(row['final_payoff/msne_eval'])

                        improvement = (msne_br - base_br) / base_br * 100.

                        if key in improvements:
                            improvements[key].append(improvement)
                        else:
                            improvements[key] = [improvement]

def removeOutliers(x, outlierConstant=1.5):
    a = np.array(x)
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    resultList = []
    for y in a:
        if y >= quartileSet[0] and y <= quartileSet[1]:
            resultList.append(y)
    return resultList

improvement_percentiles = dict()
for k in improvements.keys():
    arr = np.array(improvements[k])
    improvement_percentiles[k] = {
        'min': np.min(removeOutliers(arr)),
        'first': np.quantile(arr, 0.25),
        'median': np.quantile(arr, 0.5),
        'third': np.quantile(arr, 0.75),
        'max': np.max(removeOutliers(arr))
    }

def type_aware_sort(arr):
    try:
        return sorted(arr, key=float)
    except:
        return sorted(arr)

for g in groups:

    with open(f'multi-agent/{g}_msne_{env}.csv', 'w+') as fd:
        writer = csv.writer(fd, quoting=csv.QUOTE_NONE, escapechar=' ')
        #median box_top box_bottom whisker_top whisker_bottom name
        # writer.writerow(['index', 'name', 'whisker_bottom', 'box_bottom', 'median', 'box_top', 'whisker_top'])
        writer.writerow(['index', 'median', 'box_top', 'box_bottom', 'whisker_top', 'whisker_bottom', 'name'])
        index = 1
        for v in type_aware_sort(list(variables[groups[g]])) if groups[g] is not None else [None]:
            key = (g, v)
            writer.writerow([index, improvement_percentiles[key]['median'], improvement_percentiles[key]['first'], improvement_percentiles[key]['third'], improvement_percentiles[key]['max'], improvement_percentiles[key]['min'], v.replace(',', '-').replace(' ', '') if v is not None else g.replace('_', '-')])
            index += 1

for g in groups:
    print(f'{g},', end='')
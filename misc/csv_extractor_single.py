import csv
import numpy as np
from lazy_streams import stream

path = '/Users/taha/Desktop/TEP logs/single-agent.csv'

groups = {'layers': 'policy_params.layers',
          'batch_size': 'rl_params.batch_size',
          't_epoch': 'env_params.t_epoch',
          'act_fun': 'policy_params.act_fun',
          'gamma': 'rl_params.gamma',
          'buffer_size': 'rl_params.buffer_size',
          'history_length': 'env_params.history_length',
          'action_noise': 'training_params.action_noise_sigma',
          'env_noise': 'env_params.noise_sigma',
          'baseline': None,
          'epsilon': 'rl_params.random_exploration',
          'start_set': None
      }

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

payoff = dict()
times = dict()
for g in groups:
    for v in variables[groups[g]] if groups[g] is not None else [None]:
        key = (g, v)

        for row in rows:
            if row['Group'] == g:
                if groups[g] is None or row[groups[g]] == v:
                    p = row['base_defender_payoff']
                    t = row['Runtime']
                    if p == '':
                        continue
                    p = float(p)
                    t = int(t)
                    if key in payoff:
                        payoff[key].append(p)
                    else:
                        payoff[key] = [p]

                    if key in times:
                        times[key].append(t)
                    else:
                        times[key] = [t]


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


p_percentiles = dict()
t_percentiles = dict()
for k in payoff.keys():
    arr = np.array(payoff[k])
    p_percentiles[k] = {
        'min': np.min(removeOutliers(arr)),
        'first': np.quantile(arr, 0.25),
        'median': np.quantile(arr, 0.5),
        'third': np.quantile(arr, 0.75),
        'max': np.max(removeOutliers(arr))
    }

for k in times.keys():
    arr = np.array(times[k])
    t_percentiles[k] = {
        'min': np.min(removeOutliers(arr)),
        'first': np.quantile(arr, 0.25),
        'median': np.quantile(arr, 0.5),
        'third': np.quantile(arr, 0.75),
        'max': np.max(removeOutliers(arr))
    }

for g in groups:

    with open(f'single-agent/{g}_p.csv', 'w+') as fd:
        writer = csv.writer(fd, quoting=csv.QUOTE_NONE, escapechar=' ')
        #median box_top box_bottom whisker_top whisker_bottom name
        # writer.writerow(['index', 'name', 'whisker_bottom', 'box_bottom', 'median', 'box_top', 'whisker_top'])
        writer.writerow(['index', 'median', 'box_top', 'box_bottom', 'whisker_top', 'whisker_bottom', 'name'])
        index = 1
        for v in variables[groups[g]] if groups[g] is not None else [None]:
            key = (g, v)
            writer.writerow([index, p_percentiles[key]['median'], p_percentiles[key]['first'], p_percentiles[key]['third'], p_percentiles[key]['max'], p_percentiles[key]['min'], v.replace(',', '-').replace(' ', '') if v is not None else g.replace('_', '-')])
            index += 1

    with open(f'single-agent/{g}_t.csv', 'w+') as fd:
        writer = csv.writer(fd, quoting=csv.QUOTE_NONE, escapechar=' ')
        #median box_top box_bottom whisker_top whisker_bottom name
        # writer.writerow(['index', 'name', 'whisker_bottom', 'box_bottom', 'median', 'box_top', 'whisker_top'])
        writer.writerow(['index', 'median', 'box_top', 'box_bottom', 'whisker_top', 'whisker_bottom', 'name'])
        index = 1
        for v in variables[groups[g]] if groups[g] is not None else [None]:
            key = (g, v)
            writer.writerow([index, t_percentiles[key]['median'], t_percentiles[key]['first'], t_percentiles[key]['third'], t_percentiles[key]['max'], t_percentiles[key]['min'], v.replace(',', '-').replace(' ', '') if v is not None else g.replace('_', '-')])
            index += 1

for g in groups:
    print(f'{g},', end='')
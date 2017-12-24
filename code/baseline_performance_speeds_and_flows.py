from multiprocessing import Pool

import pandas as pd
from place_groups import selected_junctions, selected_middle_of_roads
from BaselineLinearRegressionsSpeedsAndFlows import \
    BaselineLinearRegressionSpeedsAndFlows, default_bootstrap_speed_and_flow_data_maker


def one_place(place_id):
    results = BaselineLinearRegressionSpeedsAndFlows(default_bootstrap_speed_and_flow_data_maker(place_id))\
        .baseline_errors_statistics()
    return pd.DataFrame({BaselineLinearRegressionSpeedsAndFlows.name(): results}) \
        .set_index([['speed', 'flow']]) \
        .reset_index() \
        .rename(columns={'index': 'stat'}) \
        .assign(place_id=place_id) \
        .set_index(['place_id', 'stat'])


def all_places(group, name):
    all_results = pd.concat(Pool(maxtasksperchild=1).imap_unordered(one_place, group))
    pd.concat([all_results.drop([BaselineLinearRegressionSpeedsAndFlows.name()], axis=1),
               all_results[BaselineLinearRegressionSpeedsAndFlows.name()].apply(pd.Series)], axis=1)\
        .to_csv(name + '.csv')


if __name__ == '__main__':
    all_places(selected_middle_of_roads, 'baseline_speeds_and_flows_middles')
    all_places(selected_junctions, 'baseline_speeds_and_flows_junctions')
    for name in ['baseline_speeds_and_flows_middles', 'baseline_speeds_and_flows_junctions']:
        pd.read_csv('%s.csv' % name).groupby('stat').mean().to_csv('%s_summary.csv' % name)

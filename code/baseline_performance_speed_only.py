from multiprocessing import Pool

import pandas as pd
from place_groups import selected_junctions, selected_middle_of_roads
from BaselineHistoricalAverageSpeedSingleOutput import BaselineHistoricalAverageSpeedSingleOutput
from BaselineLinearRegressionSpeedSingleOutput import BaselineLinearRegressionSpeedSingleOutput
from BaselineNaiveCopySpeedSingleOutput import BaselineNaiveCopySpeedSingleOutput
from speed_data_maker import default_bootstrap_speed_data_maker


def one_place(place_id):
    baseline_models = [
        BaselineHistoricalAverageSpeedSingleOutput,
        BaselineLinearRegressionSpeedSingleOutput,
        BaselineNaiveCopySpeedSingleOutput]
    results = {baseline_model.name():
                baseline_model(default_bootstrap_speed_data_maker(place_id, 12)).baseline_errors_statistics()
            for baseline_model in baseline_models}
    return pd.DataFrame(results)\
        .reset_index()\
        .rename(columns={'index': 'stat'})\
        .assign(place_id=place_id)\
        .set_index(['place_id', 'stat'])


def all_places(group, name):
    pd.concat(Pool(maxtasksperchild=1).imap_unordered(one_place, group)).to_csv(name + '.csv')


if __name__ == '__main__':
    all_places(selected_middle_of_roads, 'baseline_only_speed_middles')
    all_places(selected_junctions, 'baseline_only_speed_junctions')
    for name in ['baseline_only_speed_middles', 'baseline_only_speed_junctions']:
        pd.read_csv('%s.csv' % name).groupby('stat').mean().to_csv('%s_summary.csv' % name)

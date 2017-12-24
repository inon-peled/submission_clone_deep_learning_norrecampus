import os
from common import *
from place_groups import selected_junctions, selected_middle_of_roads
import functools
from collections import defaultdict
import matplotlib
import matplotlib.cm as cm
from my_gmplot import gmplot
import pandas as pd
from multiprocessing import Pool
from glob import glob


def get_avgs(place_ids):
    return pd.concat(Pool(maxtasksperchild=1).imap_unordered(one_place_since_2015, place_ids))


def get_all_place_ids():
    return [s.split('/')[-1][:-4] for s in glob(DATA_PATH + '/*.csv')]


def get_lat_lng_no_duplicates():
    LAT_LNG = pd.merge(pd.read_csv('data/../place_details.csv'),
                       pd.DataFrame().assign(place_id=get_all_place_ids()),
                       on='place_id') \
        .set_index('place_id')
    dup = defaultdict(list)
    for row in LAT_LNG.iterrows():
        dup[(row[1]['lat'], row[1]['lng'])].append((row[0], row[1]['type']))
    duplicate_place_ids = \
        functools.reduce(lambda lst1, lst2: lst1 + lst2, filter(lambda lst: len(lst) > 1, dup.values()))
    LAT_LNG_NO_DUPS = LAT_LNG.loc[~LAT_LNG.index.isin(map(lambda pair: pair[0], duplicate_place_ids))]
    assert len(LAT_LNG_NO_DUPS) == len(LAT_LNG) - len(duplicate_place_ids)
    return LAT_LNG_NO_DUPS


def rgb(vmin, vmax, value):
    m = cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin, vmax), cmap=cm.bwr)
    return '#%02X%02X%02X' % tuple(int(255 * e) for e in m.to_rgba(value)[:-1])


def rgb_a(vmin, vmax, value):
    m = cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin, vmax), cmap=cm.bwr)
    return tuple(int(255 * e) for e in m.to_rgba(value)[:-1]) + (1,)


def one_place_since_2015(place_id):
    return pd.read_csv(os.path.join(DATA_PATH, place_id + '.csv'))\
        [lambda df: df.start_interval_s >= '2015-01-01']\
        [['place_id', 'flow_bucket', 'speed_mean']]\
        .groupby('place_id')\
        .agg(['mean', 'std'])


def draw_just_location_on_map(places, places_type, radius, show_place_ids):
    norrecampus_center = 55.697731, 12.558122
    gmap = gmplot.GoogleMapPlotter(*norrecampus_center, 14)
    for row in map(lambda r: r[1], get_lat_lng_no_duplicates()[lambda df: df.index.isin(places)].iterrows()):
        gmap.circle(row.lat, row.lng, radius, 'yellow', ew=2)
        gmap.marker(row.lat, row.lng, color='red' if row.name in selected_middle_of_roads else 'yellow',
                    label=row.name[-4:] if show_place_ids else '')
    gmap.draw('%s.html' % places_type)


def draw_stat_on_map(places, places_type, stat_name, data_feature, radius):
    norrecampus_center = 55.697731, 12.558122
    gmap = gmplot.GoogleMapPlotter(*norrecampus_center, 14)
    avg_flows = get_avgs(places)[data_feature]
    unified = pd.merge(pd.DataFrame(avg_flows), get_lat_lng_no_duplicates(), left_index=True, right_index=True, how='inner').reset_index()
    for row in map(lambda r: r[1], unified.iterrows()):
        gmap.circle(row.lat, row.lng, radius,
                    rgb(min(unified[stat_name]), max(unified[stat_name]), row[stat_name]),ew=2)
        if row.place_id in selected_junctions:
            gmap.circle(row.lat, row.lng, 1, 'white',ew=2)
    gmap.draw('%s_%s_%s.html' % (data_feature.split('_')[0], places_type, stat_name))


# TODO: Add legends

if __name__ == '__main__':
    draw_just_location_on_map(selected_junctions + selected_middle_of_roads, 'middle_and_junction', 5, False)
    draw_just_location_on_map(selected_junctions, 'junction', 5, True)
    draw_just_location_on_map(selected_middle_of_roads, 'middle', 5, True)
    # draw_stat_on_map(selected_middle_of_roads, 'middle', 'mean', 'speed_mean', 20)

    # print(get_avgs(selected_middle_of_roads).reset_index().mean())
    # print(get_avgs(selected_junctions).reset_index().mean())

    # draw_on_map(get_all_place_ids(), 'all', 'mean', 'speed_mean', 5)
    # draw_on_map(get_all_place_ids(), 'all', 'std', 'speed_mean', 5)
    # draw_on_map(get_all_place_ids(), 'all', 'mean', 'flow_bucket', 5)
    # draw_on_map(get_all_place_ids(), 'all', 'std', 'flow_bucket', 5)

    # draw_on_map(selected_junctions + selected_middle_of_roads, 'mean', 'speed_mean')
    # draw_on_map(selected_junctions + selected_middle_of_roads, 'std', 'speed_mean')
    # draw_on_map(selected_junctions + selected_middle_of_roads, 'mean', 'flow_bucket')
    # draw_on_map(selected_junctions + selected_middle_of_roads, 'std', 'flow_bucket')

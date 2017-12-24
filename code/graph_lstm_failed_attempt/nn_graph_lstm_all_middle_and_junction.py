import random
random.seed(777)

from common import get_lat_lng
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1] if len(sys.argv) > 1 else '0'

from place_groups import selected_junctions, selected_middle_of_roads, filipe_places
from graph_lstm_multiple_places import GraphLSTMMultiplePlaces
from matplotlib import pyplot as plt

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 16

# Prevent tensorflow from allocating the entire GPU memory at once
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

from common import BEST_NUM_PAST_LAGS, BEST_LSTM_STATE_SIZE, BEST_MINI_BATCH_SIZE, BEST_NUM_EPOCHS

from multiple_places_at_once_speed_and_flow_data_maker import \
    default_bootstrap_multiple_places_at_once_speed_and_flow_data_maker


def process(filter, sym_norm, place_group_name, places_group, lstm_state_size, num_lags_back, mini_batch_size, num_epochs):
    nn = GraphLSTMMultiplePlaces(
        filter=filter,
        sym_norm=sym_norm,
        max_degree=None,
        num_places=len(places_group),
        loss='mse',
        data_maker=default_bootstrap_multiple_places_at_once_speed_and_flow_data_maker(places_group, num_lags_back),
        lstm_state_size=lstm_state_size,
        num_lags=num_lags_back,
        num_outs=len(places_group) * 2,
        optimizer='rmsprop',
        mini_batch_size=mini_batch_size,
        num_epochs=num_epochs,
        validation_split=0.2)
    model, history = nn.create_and_train_model()
    individual_pred_stats = nn.prediction_stats(model)
    open(('./res/individual_%s_graph_lstm_multiple_places_at_once.txt' % (place_group_name,)), 'w')\
        .write(str(individual_pred_stats))
    for group_name, group_places in [('middle', selected_middle_of_roads), ('junction', selected_junctions)]:
        open(('./%s_%s_graph_lstm_all_places_at_once.txt' % (place_group_name, group_name,)), 'w')\
            .write(str(individual_pred_stats[individual_pred_stats.index.isin(group_places)].mean()))


def sort_spatially(places, by):
    return list(get_lat_lng(places).sort_values(by=by).index)


if __name__ == '__main__':
    # Sanity check
    # process('filipe_places', filipe_places,
    #         BEST_LSTM_STATE_SIZE * len(filipe_places),
    #         BEST_NUM_PAST_LAGS,
    #         BEST_MINI_BATCH_SIZE,
    #         BEST_NUM_EPOCHS)

    # process('all_middle_and_junction', plcs_grp[::5], 1, 2, 512, 1)

    all_middle_and_junction = selected_junctions + selected_middle_of_roads
    random.shuffle(all_middle_and_junction)

    process(
        filter='localpool',
        sym_norm=True,
        place_group_name='all_middle_and_junction',
        places_group=all_middle_and_junction,
        lstm_state_size=len(all_middle_and_junction) * 2,
        num_lags_back=BEST_NUM_PAST_LAGS,
        mini_batch_size=BEST_MINI_BATCH_SIZE,
        num_epochs=int(BEST_NUM_EPOCHS * 1.5))

    # process(
    #     len(all_middle_and_junction) * 2,
    #     'all_middle_and_junction_sorted_by_lng', sort_spatially(all_middle_and_junction, 'lng'),
    #     BEST_LSTM_STATE_SIZE * len(all_middle_and_junction),
    #     BEST_NUM_PAST_LAGS,
    #     BEST_MINI_BATCH_SIZE,
    #     int(BEST_NUM_EPOCHS * 1.5))

    # process(
    #     len(all_middle_and_junction) * 2,
    #     'all_middle_and_junction_sorted_by_lat', sort_spatially(all_middle_and_junction, 'lat'),
    #     BEST_LSTM_STATE_SIZE * len(all_middle_and_junction),
    #     BEST_NUM_PAST_LAGS,
    #     BEST_MINI_BATCH_SIZE,
    #     int(BEST_NUM_EPOCHS * 1.5))

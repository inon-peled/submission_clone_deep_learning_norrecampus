import scipy.sparse as sp
from place_groups import all_places_graph, selected_middle_of_roads, selected_junctions

from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from kegra.kegra_utils import preprocess_adj, normalize_adj, normalized_laplacian, rescale_laplacian, chebyshev_polynomial
from kegra.kegra_graph import GraphConvolution

from keras.callbacks import ModelCheckpoint
import numpy as np
from generic_nn import GenericNN

from keras.layers.core import Dense
from keras.models import Sequential


def place_id_to_node_id_mapping():
    return dict(map(lambda pair: (pair[1][-4:].lower(), pair[0]),
                    enumerate(selected_middle_of_roads + selected_junctions)))


def get_adj_mat_all_middle_and_junction():
    mapping = place_id_to_node_id_mapping()
    d_arr, i_arr, j_arr = [], [], []
    for i, lst in all_places_graph.items():
        for j in lst:
            d_arr.append(1)
            d_arr.append(1)
            j_arr.append(mapping[j.lower()])
            i_arr.append(mapping[i.lower()])
            j_arr.append(mapping[i.lower()])
            i_arr.append(mapping[j.lower()])
    return sp.coo_matrix((d_arr, (i_arr, j_arr)), shape=(len(mapping), len(mapping)), dtype=np.float32)


def localpool_graph_maker(x, adj_mat, sym_norm):
    return [x, preprocess_adj(adj_mat, sym_norm)]


def chebyshev_graph_maker(x, adj_mat, sym_norm, max_degree):
    return [x] + chebyshev_polynomial(rescale_laplacian(normalized_laplacian(adj_mat, sym_norm)), max_degree)


class GraphLSTMMultiplePlaces(GenericNN):
    def __init__(self, filter, sym_norm, max_degree, num_places, *args, **kwargs):
        GenericNN.__init__(self, *args, **kwargs)
        self.filter = filter
        self.sym_norm = sym_norm
        self.max_degree = max_degree
        self.num_places = num_places

    def build_model(self):
        support = 1 if self.filter == 'localpool' else self.max_degree + 1
        G = [Input(shape=(self.num_places, None), batch_shape=(None, None), sparse=True) for _ in range(support)]
        X_in = Input(shape=(self.num_lags * 2, self.num_places, 1, 1))
        model = Sequential()
        model.add(GraphConvolution(16, support, activation='relu', W_regularizer=l2(5e-4))([X_in] + G))
        model.add(Dense(units=2, activation="linear", input_shape=(None, self.num_places * 2)))
        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def prediction_stats(self, model):
        _, _, x_test, y_test = self.data_maker.get_train_and_test_inputs()
        predictions = model.predict(np.expand_dims(np.expand_dims(np.swapaxes(x_test, 1, 2), 3), 4))
        return self.data_maker.stats(np.squeeze(predictions[:, :, :, 0::2]),
                                     np.squeeze(predictions[:, :, :, 1::2]))

    def create_and_train_model(self):
        checkpoint_path = './hdf5/%d.hdf5' % self.unique_id
        print('Creating model %s' % checkpoint_path)
        model = self.build_model()
        model.summary()
        adj_mat = get_adj_mat_all_middle_and_junction()
        x_train, y_train, _, _ = self.data_maker.get_train_and_test_inputs()
        graph = localpool_graph_maker(x_train, adj_mat, self.sym_norm) if self.filter == 'localpool' else \
            chebyshev_graph_maker(x_train, adj_mat, self.sym_norm, self.max_degree)
        # Single training iteration (we mask nodes without labels for loss calculation)
        history = model.fit(graph,
                            y_train,
                            # sample_weight=train_mask,
                            batch_size=self.mini_batch_size,
                            epochs=1,
                            shuffle=False,
                            verbose=2,
                            callbacks=[ModelCheckpoint(
                                checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')],
                            validation_split=0.2
                            )
        model.load_weights(checkpoint_path)
        # self.plot_model_training(history)
        return model, history

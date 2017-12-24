import os
from multiprocessing import Pool


def run_multiple(place_id):
    print('---------------------------------- %s -------------------------------------' % place_id)
    from rnn_class_one_segment_flow_and_speed import DefaultBootstrapper
    learner = DefaultBootstrapper().create_learner(place_id)
    learner.create_and_train_model()
    print(learner.predict())
    print(learner.baseline_lr())


if __name__ == '__main__':
    if not os.path.exists('./output/'):
        os.makedirs('./output/')
    place_ids = list(map(lambda s: s.replace('.csv', ''), os.listdir(os.path.join('.', 'data', 'by_place_5min'))))
    print(place_ids)
#     place_id = '''ChIJ4QuVTlZSUkYRRDRPcHdYULQ
# ChIJbcDEbFZSUkYRcnQFsIj5j5U
# ChIJBTt5RlZSUkYR_SyA8BgiwaM
# ChIJf9Y9sFdSUkYRmaDEJhCweGc
# ChIJj1RhMlhSUkYRxwx00g4P0QE
# ChIJozaGTFZSUkYRNtWl2AGUPkI
# ChIJP6TdhFdSUkYRdrsWKXZMAs8
# ChIJuYkcKlhSUkYRFPCipW5rTvU
# ChIJZaR1M1hSUkYRxP0WkwYYy_k'''.split('\n')
    Pool(maxtasksperchild=1, processes=32).map(run_multiple, place_ids)

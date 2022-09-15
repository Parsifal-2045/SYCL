from pickletools import float8
import sklearn.datasets as datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
from operator import concat
from numpy.random import seed
from numpy.random import randint
from time import perf_counter
import multiprocessing
from multiprocessing import Process, Manager

def produce_event(lst):
    num_layers = 100
    x_list = []
    y_list = []
    layer_list = []
    weight_list = []
    for j in range(num_layers):
        num_clusters = int(np.random.normal(loc=500, scale=15))
        for k in range(num_clusters):
            ppc = randint(8,12)
            sigma = rnd.uniform(2, 4)
            clus,_ = datasets.make_blobs(n_samples=ppc, n_features=2, centers=1, cluster_std=sigma, center_box=[-250, 250])
            x_list += clus[:,0].tolist()
            y_list += clus[:,1].tolist()
            layer_list += [j for l in range(ppc)]
            weight_list += [1 for w in range(ppc)]
    event = pd.DataFrame(list(zip(x_list, y_list, layer_list, weight_list)), columns=['x', 'y', 'layer', 'weight'])
    lst.append(event)

def write_csv(df, filename):
    df.to_csv(filename)

if __name__ =="__main__":
    num_reps = 20
    num_threads = 50
    manager = Manager()
    event_list = manager.list([])
    start_time = perf_counter()
    for r in range(num_reps):
        jobs = []
        for i in range(num_threads):
            jobs.append(multiprocessing.Process(target=produce_event, args=(event_list,)))

        for j in jobs:
            j.start()

        for j in jobs:
            j.join()

    events = pd.concat(event_list, ignore_index=True) 
    end_time = perf_counter()
    print(f'It took {end_time-start_time: 0.2f} seconds to create data for {num_threads * num_reps} events with {num_threads} threads')
    #start_to_csv = perf_counter()
    #events.to_csv("test_2_events.csv", header=False, index=True)
    #end_to_csv = perf_counter()
    #print(f'It took {end_to_csv-start_to_csv: 0.2f} seconds to create csv file for {num_threads * num_reps} events')
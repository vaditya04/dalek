from dalek.fitter import BaseFitter, BaseOptimizer
import dalek.parallel.parameter_collection as pc
import time
import numpy as np
from IPython.parallel import Client, interactive
from tardis.io.config_reader import ConfigurationNameSpace
import os
import dalek
import logging
import copy
from dalek.parallel.util import set_engines_cpu_affinity
from dalek.parallel import launcher

logger = logging.getLogger(__name__)
            
class GridBuilder(BaseOptimizer):

    def __init__(self, x_bounds, y_bounds, no_sets_in_collection):

        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
    
        self.no_sets_in_collection = no_sets_in_collection

    def __call__(self, param_collection):

        if np.any(param_collection['dalek.fitness'] == np.nan):
            raise ValueError

        max_idx = param_collection['dalek.fitness'].argmax()
        param_collection['dalek.fitness'] = np.nan
        new_x = np.random.uniform(*self.x_bounds)
        new_y = np.random.uniform(*self.y_bounds)

        param_collection.ix[max_idx] = [new_x, new_y, np.nan]

        return param_collection

    def init_parameter_collection(self):
        x_params = np.random.uniform(self.x_bounds[0], self.x_bounds[1],
                                     self.no_sets_in_collection)

        y_params = np.random.uniform(self.y_bounds[0], self.y_bounds[1],
                                     self.no_sets_in_collection)

        return dict([("x",x_params),("y",y_params)])

class TestGridRunner(object):

    def run(self, config_dict, atom_data):
        np.random.seed(250880)
        print "Building O and Si abundance arrays of 4 length"
        self.grid_builder = GridBuilder((0, 0.005), (0.2, 0.6), 4)
        self.grid_params= self.grid_builder.init_parameter_collection()
        self.config_dicts = [copy.deepcopy(config_dict) for k in range(16)]
        temp_dict = {"O" : 0, "Si" : 0}
        self.param_dicts = [copy.deepcopy(temp_dict) for k in range(16)]
        print "Assigning abundance values to parameter dictionaries:"
        for i in range(4):
            for j in range(4):
                self.param_dicts[4*i+j]["O"] = self.grid_params["x"][i]
                self.param_dicts[4*i+j]["Si"] = self.grid_params["y"][j]
                self.config_dicts[4*i+j]['model']['abundances'] = pc.apply_dict(config_dict['model']['abundances'], self.param_dicts[4*i+j])
        print "Instantiating BaseLauncher"
        rc = Client()
        print rc.ids
        blauncher = launcher.BaseLauncher(remote_clients = rc, atom_data = atom_data)
        runners = blauncher.queue_parameter_set_list(self.config_dicts, atom_data)
        try:
            runners.wait()
        except Exception, e:
            print 'Completed run'
            time.sleep(1)
            runners.display_outputs()
        return runners
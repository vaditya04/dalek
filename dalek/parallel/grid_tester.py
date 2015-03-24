from dalek.fitter import BaseFitter, BaseOptimizer
from dalek.fitter import FitterConfiguration
import dalek.parallel.parameter_collection as pc
import time
import numpy as np
from collections import OrderedDict
import pytest
from IPython.parallel import Client, interactive
from tardis.io.config_reader import ConfigurationNameSpace
import os
import dalek
import logging
import copy
from dalek.parallel.util import set_engines_cpu_affinity

try:
    from tardis import run_tardis
except ImportError:
    logger.critical('OLD version of tardis used please upgrade')
    run_tardis = lambda x: x

logger = logging.getLogger(__name__)

@interactive
def simple_worker(config_dict, atom_data=None):
    """
    This is a simple TARDIS worker that will run TARDIS and return the model

    Parameters
    ----------

    config_dict: ~dict
        a valid TARDIS config dictionary

    """
    if atom_data is None:
        if default_atom_data is None:
            raise ValueError('AtomData not available - please specify')
        else:
            atom_data = default_atom_data

    tardis_config = config_reader.Configuration.from_config_dict(
        config_dict, atom_data=atom_data)

    radial1d_mdl = model.Radial1DModel(tardis_config)
    simulation.run_radial1d(radial1d_mdl)
    return radial1d_mdl


class BaseLauncher(object):
    """
    The base class of the the launcher to launch groups of parameter sets and
    evaluate them on remote machines

    Parameters
    ----------

    remote_clients: ~IPython.parallel.Client
        IPython remote clients

    worker: func
        a function pointer to the worker function [default=simple_worker]

    atom_data: ~tardis.atomic.AtomData
        an atom_data instance that is copied to all the remote clients
        if None, each time an atom_data needs to be pushed to the client

    """



    def __init__(self, remote_clients, worker=simple_worker,
                 atom_data=None):
        self.remote_clients = remote_clients
        self.prepare_remote_clients(remote_clients, atom_data)
        self.worker = worker
        self.lbv = remote_clients.load_balanced_view()
        self.remote_clients[:].use_dill()   #This fixes the pickling error, now map function can be used

    @staticmethod
    def prepare_remote_clients(clients, atom_data):
        """
        Preparing the remote clients for computation: Uploading the atomic
        data if available and making sure that the clients can run on different
        CPUs on each Node

        Parameters
        ----------

        clients: IPython.parallel.Client
            remote clients from ipython

        atom_data: tardis.atomic.AtomData or None
            remote atomic data, if None each queue needs to bring their own one
        """

        logger.info('Sending initial atomic dataset to remote '
                    'clients and importing tardis')
        clients.block = True
        for client in clients:
            client['default_atom_data'] = atom_data
            client.execute('from tardis.io import config_reader')
            client.execute('from tardis import model, simulation')

        clients.block = False


        for client in clients:
            client.apply(set_engines_cpu_affinity)

    def queue_parameter_set(self, parameter_set_dict, atom_data=None):
        """
        Add single parameter set to the queue

        Parameters
        ----------

        parameter_set_dict: ~dict
            a valid configuration dictionary for TARDIS
        """
        return self.lbv.apply(self.worker, parameter_set_dict, atom_data=atom_data)

    def queue_parameter_set_list(self, parameter_set_list,
                                      atom_data=None):
        """
        Add a list of parameter sets to the queue

        Parameters
        ----------

        parameter_set_dicts: ~list of ~dict
            a list of valid configuration dictionary for TARDIS
        """

        atom_dataset = [copy.deepcopy(atom_data) for k in range(len(parameter_set_list))]
        return self.lbv.map(self.worker, parameter_set_list, atom_dataset )
            
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
        blauncher = BaseLauncher(remote_clients = rc, atom_data = atom_data)
        runners = blauncher.queue_parameter_set_list(self.config_dicts, atom_data)
        try:
            runners.wait()
        except Exception, e:
            print 'Completed run'
            time.sleep(1)
            runners.display_outputs()
        return runners
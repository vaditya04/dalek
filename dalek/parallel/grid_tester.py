from dalek.fitter import BaseFitter, BaseOptimizer
from dalek.fitter import FitterConfiguration
import dalek.parallel.parameter_collection as pc
# from dalek.parallel.parameter_collection import ParameterCollection2
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

#Following are attempts to get queue_parameter_set_list working with simple_worker method. 
#Maybe they can be used to fix the problem, but for now I'm of the opinion that we need to rewrite the function
import copy_reg
import types
import pickle
from functools import wraps

try:
    from tardis import run_tardis
except ImportError:
    logger.critical('OLD version of tardis used please upgrade')
    run_tardis = lambda x: x

logger = logging.getLogger(__name__)

def _pickle_method(method):
    """
    Pickle methods properly, including class methods.
    """
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    if isinstance(cls, type):
        # handle classmethods differently
        cls = obj
        obj = None
    if func_name.startswith('__') and not func_name.endswith('__'):
        #deal with mangled names
        cls_name = cls.__name__.lstrip('_')
        func_name = '_%s%s' % (cls_name, func_name)

    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    """
    Unpickle methods properly, including class methods.
    """
    if obj is None:
        return cls.__dict__[func_name].__get__(obj, cls)
    for cls in cls.__mro__:
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

def reduce_method(m):
    return (getattr, (m.__self__, m.__func__.__name__))
#end unsucessful methods
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

#This function does not work with simple_worker right now due to it being an instance method with a decorator on top of it
    def queue_parameter_set_list(self, parameter_set_list,
                                      atom_data=None):
        """
        Add a list of parameter sets to the queue

        Parameters
        ----------

        parameter_set_dicts: ~list of ~dict
            a list of valid configuration dictionary for TARDIS
        """

        # print len(parameter_set_list)
        # config_atom_dataset = [(parameter_set_list[k],copy.deepcopy(atom_data)) for k in range(len(parameter_set_list))]
        # print config_atom_dataset
         # self.config_dicts = [copy.deepcopy(config_dict) for k in range(16)]
        # copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)
        # copy_reg.pickle(types.MethodType,reduce_method )
        # pickled_worker = pickle.loads(pickle.dumps(self.worker))

        #Temporary fix until a solution for pickling for map function is found - This method is merely
        #running apply on a loop
        atom_dataset = [copy.deepcopy(atom_data) for k in range(len(parameter_set_list))]
        # ans = []
        # for i in range(len(parameter_set_list)):
            # ans.append(self.lbv.apply(self.worker, parameter_set_list[i], atom_dataset[i]))
        # return ans
        return self.lbv.map(self.worker, atom_dataset)

    def queue_run_tardis(self, config_dict, atom_data = None):
        return self.lbv.apply(run_tardis,config_dict,atom_data)
            
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

        # initial_param_collection = ParameterCollection(
            # OrderedDict([('param.x', x_params), ('param.y', y_params)]))
        # initial_param_collection['dalek.fitness'] = np.nan
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
        runner = []
        for i in range(len(self.config_dicts)):
            runner.append(blauncher.queue_parameter_set(self.config_dicts[i], atom_data))
        for i in range(len(self.config_dicts)):
            runner[i].wait()
            runner[i].display_outputs()
import logging

from IPython.parallel import interactive, RemoteError
logger = logging.getLogger(__name__)
from dalek.parallel.util import set_engines_cpu_affinity

try:
    from tardis import run_tardis
except ImportError:
    logger.critical('OLD version of tardis used please upgrade')
    run_tardis = lambda x: x

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
        config_dict, atom_data=atom_data, validate=False)
    radial1d_mdl = model.Radial1DModel(tardis_config)
    simulation.run_radial1d(radial1d_mdl)

    return radial1d_mdl


@interactive
def fitter_worker(config_dict, atom_data=None):
    """
    This is a TARDIS worker that will run TARDIS and evaluate the returned model
    by running the pushed fitness_function object

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
        config_dict, atom_data=atom_data, validate=False)
    radial1d_mdl = model.Radial1DModel(tardis_config)
    simulation.run_radial1d(radial1d_mdl)

    fitness, spectrum = fitness_function(radial1d_mdl)

    return fitness, spectrum

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

        return self.lbv.apply(self.worker, parameter_set_dict,
                              atom_data=atom_data)

    def queue_parameter_set_list(self, parameter_set_list,
                                      atom_data=None):
        """
        Add a list of parameter sets to the queue

        Parameters
        ----------

        parameter_set_dicts: ~list of ~dict
            a list of valid configuration dictionary for TARDIS
        """

        return self.lbv.map(self.worker, parameter_set_list,
                            atom_data=atom_data)



class FitterLauncher(BaseLauncher):

    def __init__(self, remote_clients, fitness_function, atom_data=None,
                 worker=fitter_worker):
        self.fitness_function = fitness_function
        super(FitterLauncher, self).__init__(remote_clients,
                                           worker=worker,
                                           atom_data=atom_data)

    def prepare_remote_clients(self, clients, atom_data):

        super(FitterLauncher, self).prepare_remote_clients(clients, atom_data)
        clients.block = True
        for client in clients:
            client['fitness_function'] = self.fitness_function
        clients.block = False
        logger.info('Initial setup complete')

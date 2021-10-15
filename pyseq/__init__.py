from . import methods

# Functions common to all HiSeq models

def get_instrument(virtual=False):

    # Get instrument model and name
    model, name = methods.get_machine_info(args_['virtual'])

    # Create HiSeq Object
    if model == 'HiSeq2500':
        if args_['virtual']:
            from . import virtualHiSeq
            hs = virtualHiSeq.HiSeq2500(name, logger)
        else:
            from . import hiseq2500
            hs = hiseq2500.HiSeq(name, logger)
    else:
        hs = None

    return hs



def setup_logger(experiment_name=None, log_path = None, config=None):
    """Create a logger and return the handle."""

    if log_path is None and config is not None:
        log_path = config.get('experiment','log_path')
    if experiment_name is None and config is not None:
        experiment_name = config.get('experiment','experiment name')

    # Create a custom logger
    logger = logging.getLogger(__name__)
    logger.setLevel(10)

    # Create console handler
    c_handler = logging.StreamHandler()
    c_handler.setLevel(21)
    # Create file handler
    f_log_name = join(log_path,experiment_name + '.log')
    f_handler = logging.FileHandler(f_log_name)
    f_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(asctime)s - %(message)s', datefmt = '%Y-%m-%d %H:%M')
    f_format = logging.Formatter('%(asctime)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger

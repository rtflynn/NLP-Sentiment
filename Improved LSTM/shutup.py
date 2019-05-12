import os
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
sys.stderr = stderr


# The following is to get rid of annoying tensorflow msgs regarding deprecation
# and gpu setup and so on

def tensorflow_shutup():
    """
    Make Tensorflow less verbose
    """
    try:
        # noinspection PyPackageRequirements
        import os
        from tensorflow import logging
        logging.set_verbosity(logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        def deprecated(date, instructions, warn_once=True):
            def deprecated_wrapper(func):
                return func
            return deprecated_wrapper

        from tensorflow.python.util import deprecation
        deprecation.deprecated = deprecated

    except ImportError:
        pass

tensorflow_shutup()
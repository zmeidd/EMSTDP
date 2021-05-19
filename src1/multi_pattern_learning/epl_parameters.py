# - adapted from EPL module - NxSDKapp 0.9.5rc
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from collections import namedtuple

def positiveInt(val):
    """validates if the value passed is a positive integer or not"""
    if val < 1 or type(val) != int:
        raise ValueError("value passed must be a positive integer")
    else:
        return val


def prob(val):
    """ validates the range of (connection) probability"""
    if not (type(val) == float and 0.0 <= val <= 1.0):
        raise ValueError("0.0 <= val <= 1.0 is the correct range")
    else:
        return val


def randSeed(val):
    """ validates the range of random seed"""
    if type(val) == int and 0 <= val < 2**32:
        return val
    else:
        raise ValueError("0 <= random seed < 2^32 is the correct range")


def binary(val):
    """ validates if the value passed is binary (true/false)"""
    if type(val) == bool:
        return val
    else:
        raise ValueError("random seed is a boolean flag")


# Each parameter is a named tuple of (parameter name, default value and its
# associated validator
Parameter = namedtuple('Parameter', ['name', 'default', 'validator'])
# list of parameters
parameters = [
    # for network setup
    Parameter(name='numlayers', default=2, validator=positiveInt),

    ### MNIST 10 class (uncomment the following)
    Parameter(name='numInputs', default=32*32*3, validator=positiveInt),
    Parameter(name='numHidNurns', default=[100, 9], validator=positiveInt),
    Parameter(name='numTargets', default=9, validator=positiveInt),

    Parameter(name='phaseLength', default=32, validator=positiveInt),
    Parameter(name='connProbMCToGC', default=0.2, validator=prob),
    Parameter(name='useRandomSeed', default=False, validator=binary),
    Parameter(name='randomGenSeed', default=0, validator=randSeed),

    Parameter(name='numTrainingSamples', default=2, validator=positiveInt),
    Parameter(name='numTestingSamples', default=1, validator=positiveInt),
    # network operation
    Parameter(name='numGammaCyclesTrain', default=45, validator=positiveInt),
    Parameter(name='numGammaCyclesTest', default=5, validator=positiveInt),
    Parameter(name='numGammaCyclesIdle', default=5, validator=positiveInt),
    Parameter(name='useLMTSpikeCounters', default=False, validator=binary),
    # logistics
    Parameter(name='logSNIPs', default=False, validator=binary),
    Parameter(name='executionTimeProbe', default=True, validator=binary)
]

validators = {par.name: par.validator for par in parameters}
defaults = {par.name: par.default for par in parameters}


class ParamsEPLSlots:
    """ Defines the various parameters of the EPL network. Defined as slots
    for ease of use"""
    __slots__ = ['numlayers', 'numInputs',
                 'numHidNurns', 'numTargets', 'phaseLength', 'connProbMCToGC',
                 'useRandomSeed', 'randomGenSeed', 'numTrainingSamples', 'numTestingSamples', 'numGammaCyclesTrain',
                 'numGammaCyclesTest', 'numGammaCyclesIdle',
                 'useLMTSpikeCounters', 'logSNIPs', 'executionTimeProbe']

    def __init__(self):
        pass


class ParamemtersForEPL(ParamsEPLSlots):
    """ Sets the default values for the parameters to EPL network. The values
    of the parameter are validated every time they are set.

    :param int numlayers: number of patterns to be learned by the EPL \
    network
    :param int numInputs: number of columns in the network
    :param int numMCsPerColumn: number of MCs per column
    :param int numHidNurnsPerColumn: number of GCs per column for each \
            different pattern
    :param int numDelaysMCToGC: number of different MC->GC connnection delays
    :param int minDelaysMCToGC: value of the smallest MC->GC delay
    :param int phaseLength: number of algorithmic timesteps in a \
    gamma cycle
    :param float connProbMCToGC: MC->GC connection probability
    :param bool useRandomSeed: used to set if we want to use a random seed
    :param int randomGenSeed: the value of random seed
    :param int numGammaCyclesTrain: number of gamma cycles to train the network
    :param int numGammaCyclesTest: number of gamma cycles to test the network
    :param int numGammaCyclesIdle: number of gamma cycles when no inputs are \
    presented
    :param bool useLMTSpikeCounters: set to true if we want to use SNIP based \
    spike counter instead of spike probes
    :param bool logSNIPs: used to turn SNIP logging on and off
    :param bool executionTimeProbe: set to enable/disable execution time probe

    """

    def __init__(self):
        super().__init__()
        for par in parameters:
            setattr(self, par.name, par.default)

    def __setattr__(self, key, value):
        """ Ensures that only the parameters defined in slots are set"""
        if key not in self.__slots__:
            raise \
                AttributeError(
                    """Can't create new attribute '{}'""".format(key))
        # validator = validators[key]
        # value = validator(value)
        ParamsEPLSlots.__dict__[key].__set__(self, value)

    def showDefaults(self):
        """ lists the default values for the various parameters"""
        for attr in self.__slots__:
            print("{}={}".format(attr, getattr(self, attr)))


if __name__ == '__main__':
    print([par.name for par in parameters])
    par = ParamemtersForEPL()
    par.showDefaults()


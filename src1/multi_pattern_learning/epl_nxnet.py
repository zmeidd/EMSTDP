# - adapted from EPL module - NxSDKapp 0.9.5rc
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import nxsdk.api.n2a as nx
import numpy as np
import math
import random
import time
import os
from collections import namedtuple
from .epl_parameters import ParamemtersForEPL, ParamsEPLSlots
import matplotlib.pyplot as plt

# import scipy.sparse as sp
import scipy.sparse as sps


def timer(input_func):
    """ returns the execution time of a function wrapped by this decorator"""

    def timed(*args, **kwargs):
        start_time = time.time()
        result = input_func(*args, **kwargs)
        end_time = time.time()
        print("{0} took {1:0.5f} secs".format(input_func.__name__,
                                              end_time - start_time))
        return result

    return timed


def init_wgts(wmin, wmax, wdes, wsrc, sd):
    # np.random.seed((sd+1)*10)

    tmpp = np.random.normal(0, np.sqrt(3.0 / float(wsrc)), [wdes, wsrc])

    # wgts = tmpp * wmax

    # wm = np.max(tmpp)
    # wgts = tmpp * float(wmax/wm)
    # wgts = np.clip(wgts, -255, 255)

    amx = np.max(tmpp)
    amn = np.min(tmpp)
    a1 = (tmpp - amn) / (amx - amn)
    a1 = a1 * (wmax - wmin) + wmin
    wgts = np.clip(a1, a_min=wmin, a_max=wmax)

    # wgts = np.random.randint(low=wmin, high=wmax, size=(wdes, wsrc))

    wgts = wgts.astype(int)

    return wgts


def init_th(wsrc, layer, scale, wmax):
    hThr = float(scale / (1)) * wmax * wsrc * (np.sqrt(3.0 / float(wsrc)) / (2.0))
    # hThr = float(scale) * wmax * wsrc * (np.sqrt(3.0 / float(wsrc)) / (2.0))
    #     hThr = float(scale / (layer + 1)) * wmax * wsrc * 1.0 * 0.5
    hThr = int(hThr)
    return hThr


class EplNxNet(ParamsEPLSlots):
    """NxNet implementation of the EPL network"""

    def __init__(self, eplParams):
        # make sure the type of the parameters class
        assert isinstance(eplParams, ParamemtersForEPL)
        super().__init__()
        # copy the parameters
        for attr in eplParams.__slots__:
            setattr(self, attr, getattr(eplParams, attr))

        self.net = nx.NxNet()

        self.stim2bias = [int(1) for i in range(1)]
        self.stim2bias += [int(i * 1) for i in range(1, 256, 1)]

        self.numlayers = len(self.numHidNurns)
        self.numMCs = self.numInputs
        self.numGCs = np.sum(self.numHidNurns)

        self.convNumInputs = 0

        self.poswgtrng = 64
        self.negwgtrng = -64

        self.bposwgtrng = 128
        self.bnegwgtrng = -128

        # probes related data structures
        self.allMCSomaProbes = None
        self.exc2InhConnProbes = None
        self.inh2ExcConnProbesPos = None
        self.inh2ExcConnProbesNeg = None
        self.mcADProbes = None
        self.mcSomaProbes = None
        self.gcProbes = None
        self.numStepsRan = 0

    def setupNetwork(self, train, conv_wgt, wgt, bwgt):
        """ setups the EPL network """

        if train:
            self.trainbool = 1
        else:
            self.trainbool = 0

        self.isConv = False

        self.allGCsPerPattern = dict()
        self.allPosECsPerPattern = dict()
        self.allNegECsPerPattern = dict()
        self.allTmpPosECsPerPattern = dict()
        self.allTmpNegECsPerPattern = dict()

        self.forwardConns = dict()
        self.posbackwardConns = dict()
        self.negbackwardConns = dict()
        self.lastUsedLogicalCoreId = 0

        # self.allMCSomaGrp = self.net.createCompartmentGroup()
        self.allMCSomaGrp = self.net.createNeuronGroup()
        self.allLabelGrp = self.net.createNeuronGroup()
        self.wtaGrp = self.net.createCompartmentGroup()

        self.createMCNeurons()

        self.gcCoreIdRange = [self.lastUsedLogicalCoreId + 1]

        self.allMCSomaGrp.soma.sizeX = int(np.sqrt(self.numMCs))
        self.allMCSomaGrp.soma.sizeY = int(np.sqrt(self.numMCs))
        self.allMCSomaGrp.soma.sizeC = 1



        ### create convolutional layers #must match the conv layers in the saved keras model
        if conv_wgt:
            self.convSpec = []
            self.createConvLayers(conv_wgt)
            # self.createConvLayers1(conv_wgt)
            # self.createConvLayers2(conv_wgt)

        self.gcCoreIdRange.append(self.lastUsedLogicalCoreId + 1)

        for patternIdx in range(self.numlayers):
            self.allGCsPerPattern[patternIdx] = self.net.createCompartmentGroup()
            if patternIdx != self.numlayers - 1:
                self.allPosECsPerPattern[patternIdx] = self.net.createNeuronGroup()
                self.allNegECsPerPattern[patternIdx] = self.net.createNeuronGroup()
                self.allTmpPosECsPerPattern[patternIdx] = self.net.createCompartmentGroup()
                self.allTmpNegECsPerPattern[patternIdx] = self.net.createCompartmentGroup()
            else:
                self.allPosECsPerPattern[patternIdx] = self.net.createCompartmentGroup()
                self.allNegECsPerPattern[patternIdx] = self.net.createCompartmentGroup()
                self.allTmpPosECsPerPattern[patternIdx] = self.net.createCompartmentGroup()
                self.allTmpNegECsPerPattern[patternIdx] = self.net.createCompartmentGroup()

            self.createGCNeuronsPerPattern(patternIdx)

        self.connectforwardConns(train, wgt)
        self.connectbackwardConns(bwgt)
        self.gcCoreIdRange.append(self.lastUsedLogicalCoreId)

    def createMCNeurons(self, biasMant=0):
        """ configures  the MC neurons"""
        tauU = 2
        tauV = 25
        decayU = int(1 / tauU * 2 ** 12)
        decayV = int(1 / tauV * 2 ** 12)
        decayU = 4095
        decayV = 0
        vth = 256
        inWgt = 35  # Input spike connection weight
        self.dtrite = []

        maxColsPerCore = 200

        for colIdx in range(self.numInputs):
            coreIdx = self.lastUsedLogicalCoreId + \
                      math.ceil((colIdx + 1) / maxColsPerCore)
            # mcSomaProto = nx.CompartmentPrototype(
            #     logicalCoreId=coreIdx,
            #     compartmentCurrentDecay=0,
            #     compartmentVoltageDecay=0,
            #     biasExp=0,
            #     #                 enableSpikeBackprop=1,
            #     #                 enableSpikeBackpropFromSelf=1,
            #     vThMant=4 * 4,  # i.e. 2 * 64 = 128
            #     refractoryDelay=1,
            #     vMinExp=0,
            #     # enableHomeostasis=0,
            #     # numDendriticAccumulators=64,
            #     # enableNoise=1,
            #     # randomizeVoltage=1,
            #     # noiseMantAtCompartment=5,
            #     # noiseExpAtCompartment=5,
            #     functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
            #     thresholdBehavior=nx.COMPARTMENT_THRESHOLD_MODE.SPIKE_AND_RESET
            # )
            #
            # mcSomaCx = self.net.createCompartment(prototype=mcSomaProto)
            # self.allMCSomaGrp.addCompartments(mcSomaCx)

            # create a neuron prototype np0
            np0 = nx.NeuronPrototypes.NeuronSoftResetTwoCompartments(decayU, decayV, vth, logicalCoreId=coreIdx)

            # create a two-compartment neuron with prototype np0
            neuron0 = self.net.createNeuron(np0)

            # Connect somatic compartment with dendritic compartment
            wgtInh = -int(vth * decayU / 4096)
            connProto1 = nx.ConnectionPrototype(weight=wgtInh)
            neuron0.soma.connect(neuron0.dendrites[0], connProto1)
            neuron0.dendrites[0].biasExp = 6
            self.allMCSomaGrp.addNeuron(neuron0)

        # self.dtrite = self.allMCSomaGrp.dendrites[0].nodeIds

        self.lastUsedLogicalCoreId += math.ceil(self.numInputs / maxColsPerCore)
        # print(self.lastUsedLogicalCoreId)

    # def createMCNeurons(self, biasMant=0):
    #     """ configures  the MC neurons"""
    #     maxColsPerCore = 200
    #     # for colIdx in range(self.numInputs):
    #     #     coreIdx = self.lastUsedLogicalCoreId + \
    #     #               math.ceil((colIdx + 1) / maxColsPerCore)
    #     #     mcSomaProto = nx.CompartmentPrototype(
    #     #         logicalCoreId=coreIdx,
    #     #         compartmentCurrentDecay=0,
    #     #         compartmentVoltageDecay=0,
    #     #         biasExp=0,
    #     #         #                 enableSpikeBackprop=1,
    #     #         #                 enableSpikeBackpropFromSelf=1,
    #     #         vThMant=4 * 4,  # i.e. 2 * 64 = 128
    #     #         refractoryDelay=1,
    #     #         vMinExp=0,
    #     #         # enableHomeostasis=0,
    #     #         # numDendriticAccumulators=64,
    #     #         # enableNoise=1,
    #     #         # randomizeVoltage=1,
    #     #         # noiseMantAtCompartment=5,
    #     #         # noiseExpAtCompartment=5,
    #     #         functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
    #     #         thresholdBehavior=nx.COMPARTMENT_THRESHOLD_MODE.SPIKE_AND_RESET
    #     #     )
    #     #
    #     #     mcSomaCx = self.net.createCompartment(prototype=mcSomaProto)
    #     #     self.allMCSomaGrp.addCompartments(mcSomaCx)
    #
    #     tauU = 2
    #     tauV = 25
    #     # decayU = int(1 / tauU * 2 ** 12)
    #     # decayV = int(1 / tauV * 2 ** 12)
    #     decayU = 4095
    #     decayV = 0
    #     vth = 256
    #     inWgt = 35  # Input spike connection weight
    #
    #     # create compartment prototypes for two-compartment soft-reset neurons
    #     cxProtoDendrite = nx.CompartmentPrototype(compartmentCurrentDecay=decayU,
    #                                               compartmentVoltageDecay=decayV,
    #                                               thresholdBehavior=nx.COMPARTMENT_THRESHOLD_MODE.NO_SPIKE_AND_PASS_V_LG_VTH_TO_PARENT,
    #                                               vThMant=vth,
    #                                               functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
    #                                               # biasMant=args.bias,
    #                                               biasExp=6,
    #                                               refractoryDelay=1,
    #                                               )
    #     # logicalCoreId=0)
    #     cxProtoSoma = nx.CompartmentPrototype(compartmentCurrentDecay=4095,
    #                                           compartmentVoltageDecay=0,
    #                                           vThMant=vth, )
    #     # logicalCoreId=0)
    #     cxProtoSoma.addDendrite(cxProtoDendrite, nx.COMPARTMENT_JOIN_OPERATION.OR)
    #
    #     # create 2 seperate compartment groups with previously defined prototypes
    #     # net = nx.NxNet()
    #     self.allMCDendriteGrp = self.net.createCompartmentGroup(size=self.numInputs, prototype=cxProtoDendrite)
    #     self.allMCSomaGrp = self.net.createCompartmentGroup(size=self.numInputs, prototype=cxProtoSoma)
    #
    #     corenum = 0
    #     for compa, compb in zip(self.allMCDendriteGrp, self.allMCSomaGrp):
    #         compa.logicalCoreId = int(np.floor(corenum))
    #         compb.logicalCoreId = int(np.floor(corenum))
    #         corenum = corenum + 1 / maxColsPerCore
    #
    #     # self.dtrite = []
    #     # setup input compartment for each soma to form multi-compartment neurons
    #     for idx in range(self.numInputs):
    #         self.allMCSomaGrp[idx].cxGroupinputCompartmentId0 = self.allMCDendriteGrp[idx].nodeId
    #         # self.dtrite.append(self.allMCDendriteGrp[idx].nodeId)
    #
    #     # specify feedback connection prototype
    #     connProto = nx.ConnectionPrototype(signMode=nx.SYNAPSE_SIGN_MODE.MIXED,
    #                                        # compressionMode=nx.SYNAPSE_COMPRESSION_MODE.SPARSE,
    #                                        # numTagBits=0,
    #                                        # delay=0,
    #                                        # numDelayBits=0,
    #                                        # enableDelay=0
    #                                        )
    #
    #     # make 1-1 connection between dendrite and soma
    #     maskInh = sps.eye(self.numInputs, dtype=int).tocoo()
    #     wgtInh = maskInh.copy() * -vth
    #     # wgtInh = maskInh.copy() * -int(vth * decayU / 4096)
    #     self.allMCSomaGrp.connect(self.allMCDendriteGrp, prototype=connProto, connectionMask=maskInh, weight=wgtInh)
    #
    #
    #
    #     self.lastUsedLogicalCoreId += math.ceil(self.numInputs / maxColsPerCore)
    #     # print(self.lastUsedLogicalCoreId)

    def createConvLayers(self, conv_wgt):

        self.isConv = True

        biasEx = 0  # bias exponential
        biasMn = 1  # bias mantissa default
        self.convTh = 0.3

        def get_convcompProto(vTh):
            convProto = nx.CompartmentPrototype(
                # logicalCoreId=coreIdx,
                compartmentCurrentDecay=4095,
                compartmentVoltageDecay=0,
                # biasExp=0,
                biasMant=biasMn,
                biasExp=biasEx,
                vThMant=vTh,  # i.e. 2 * 64 = 128
                refractoryDelay=1,
                vMinExp=20,
                # enableHomeostasis=0,
                # numDendriticAccumulators=64,

                # enableNoise=1,
                # randomizeVoltage=1,
                # noiseMantAtCompartment=5,
                # noiseExpAtCompartment=6,

                functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
                thresholdBehavior=nx.COMPARTMENT_THRESHOLD_MODE.SPIKE_AND_RESET
            )
            return convProto

        def get_poolcompProto(vTh):
            poolProto = nx.CompartmentPrototype(
                # logicalCoreId=coreIdx,
                compartmentCurrentDecay=4095,
                compartmentVoltageDecay=0,
                # biasExp=0,
                biasMant=biasMn,
                biasExp=biasEx,
                vThMant=vTh,  # i.e. 2 * 64 = 128
                refractoryDelay=1,
                vMinExp=20,
                # enableHomeostasis=0,
                # numDendriticAccumulators=64,
                # enableNoise=1,
                # randomizeVoltage=1,
                # noiseMantAtCompartment=5,
                # noiseExpAtCompartment=5,
                functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
                thresholdBehavior=nx.COMPARTMENT_THRESHOLD_MODE.SPIKE_AND_RESET
            )
            return poolProto

        convSpec = dict()
        convSpec["dimX"] = 3
        convSpec["dimY"] = 3
        convSpec["dimC"] = 16
        convSpec["stride"] = 2
        convSpec["weightFile"] = conv_wgt[0]
        wsrc = convSpec["dimX"] * convSpec["dimY"] * 1
        vTh = init_th(wsrc, 1, self.convTh, 255)
        convSpec["compProto"] = get_convcompProto(vTh)
        compartmentsPerCore = 4096 / 8
        self.convSpec.append(convSpec)

        self.layer1, corenum = self.convLayer(self.allMCSomaGrp.soma, convSpec, self.lastUsedLogicalCoreId + 1,
                                              compartmentsPerCore)

        self.lastUsedLogicalCoreId = int(np.ceil(corenum))

        # poolSpec = dict()
        # poolSpec["stride"] = 3
        # poolSpec["compProto"] = poolProto
        # # poolSpec["weightFile"] = np.ones((poolSpec["stride"], poolSpec["stride"]), dtype=int) * 28*4
        # poolSpec["weightFile"] = int(28*(4/poolSpec["stride"]))
        # compartmentsPerCore = 4096 / 16
        #
        # self.layer2, corenum = self.poolingLayer(self.layer1, poolSpec, self.lastUsedLogicalCoreId+1, compartmentsPerCore)
        #
        # self.lastUsedLogicalCoreId = int(np.ceil(corenum))

        convSpec = dict()
        convSpec["dimX"] = 3
        convSpec["dimY"] = 3
        convSpec["dimC"] = 8
        convSpec["stride"] = 2
        convSpec["weightFile"] = conv_wgt[1]
        wsrc = convSpec["dimX"] * convSpec["dimY"] * self.convSpec[0]["dimC"]
        vTh = init_th(wsrc, 1, self.convTh, 255)
        convSpec["compProto"] = get_convcompProto(vTh)
        compartmentsPerCore = 4096 / 8
        self.convSpec.append(convSpec)

        self.layer2, corenum = self.convLayer(self.layer1, convSpec, self.lastUsedLogicalCoreId + 1,
                                              compartmentsPerCore)

        self.lastUsedLogicalCoreId = int(np.ceil(corenum))

        for i, cspec in enumerate(self.convSpec):
            print("conv_layer_", i)
            print("dimX", cspec["dimX"], "dimY", cspec["dimY"], "dimC", cspec["dimC"], "stride", cspec["stride"], "vth",
                  cspec["compProto"].vThMant)

        # re-order the compartment group to SlayerPyTorch convention before the fully connected layers
        self.lastConvLayer = self.reorderLayer(self.layer2)

        self.convNumInputs = self.lastConvLayer.numNodes

    def createConvLayers1(self, conv_wgt):

        self.isConv = True

        biasEx = 0  # bias exponential
        biasMn = 1  # bias mantissa default
        self.convTh = 0.3

        def get_convcompProto(vTh):
            convProto = nx.CompartmentPrototype(
                # logicalCoreId=coreIdx,
                compartmentCurrentDecay=4095,
                compartmentVoltageDecay=0,
                # biasExp=0,
                biasMant=biasMn,
                biasExp=biasEx,
                vThMant=vTh,  # i.e. 2 * 64 = 128
                refractoryDelay=1,
                vMinExp=20,
                # enableHomeostasis=0,
                # numDendriticAccumulators=64,

                # enableNoise=1,
                # randomizeVoltage=1,
                # noiseMantAtCompartment=5,
                # noiseExpAtCompartment=6,

                functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
                thresholdBehavior=nx.COMPARTMENT_THRESHOLD_MODE.SPIKE_AND_RESET
            )
            return convProto

        def get_poolcompProto(vTh):
            poolProto = nx.CompartmentPrototype(
                # logicalCoreId=coreIdx,
                compartmentCurrentDecay=4095,
                compartmentVoltageDecay=0,
                # biasExp=0,
                biasMant=biasMn,
                biasExp=biasEx,
                vThMant=vTh,  # i.e. 2 * 64 = 128
                refractoryDelay=1,
                vMinExp=20,
                # enableHomeostasis=0,
                # numDendriticAccumulators=64,
                # enableNoise=1,
                # randomizeVoltage=1,
                # noiseMantAtCompartment=5,
                # noiseExpAtCompartment=5,
                functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
                thresholdBehavior=nx.COMPARTMENT_THRESHOLD_MODE.SPIKE_AND_RESET
            )
            return poolProto

        convSpec = dict()
        convSpec["dimX"] = 3
        convSpec["dimY"] = 3
        convSpec["dimC"] = 8
        convSpec["stride"] = 2
        convSpec["weightFile"] = conv_wgt[0]
        wsrc = convSpec["dimX"] * convSpec["dimY"] * 1
        vTh = init_th(wsrc, 1, self.convTh, 255)
        convSpec["compProto"] = get_convcompProto(vTh)
        compartmentsPerCore = 4096 / 8
        self.convSpec.append(convSpec)

        self.layer1, corenum = self.convLayer(self.allMCSomaGrp.soma, convSpec, self.lastUsedLogicalCoreId + 1,
                                              compartmentsPerCore)

        self.lastUsedLogicalCoreId = int(np.ceil(corenum))

        # poolSpec = dict()
        # poolSpec["stride"] = 3
        # poolSpec["compProto"] = poolProto
        # # poolSpec["weightFile"] = np.ones((poolSpec["stride"], poolSpec["stride"]), dtype=int) * 28*4
        # poolSpec["weightFile"] = int(28*(4/poolSpec["stride"]))
        # compartmentsPerCore = 4096 / 16
        #
        # self.layer2, corenum = self.poolingLayer(self.layer1, poolSpec, self.lastUsedLogicalCoreId+1, compartmentsPerCore)
        #
        # self.lastUsedLogicalCoreId = int(np.ceil(corenum))

        convSpec = dict()
        convSpec["dimX"] = 3
        convSpec["dimY"] = 3
        convSpec["dimC"] = 8
        convSpec["stride"] = 2
        convSpec["weightFile"] = conv_wgt[1]
        wsrc = convSpec["dimX"] * convSpec["dimY"] * self.convSpec[0]["dimC"]
        vTh = init_th(wsrc, 1, self.convTh, 255)
        convSpec["compProto"] = get_convcompProto(vTh)
        compartmentsPerCore = 4096 / 8
        self.convSpec.append(convSpec)

        self.layer2, corenum = self.convLayer(self.layer1, convSpec, self.lastUsedLogicalCoreId + 1,
                                              compartmentsPerCore)

        self.lastUsedLogicalCoreId = int(np.ceil(corenum))

        for i, cspec in enumerate(self.convSpec):
            print("conv_layer_", i)
            print("dimX", cspec["dimX"], "dimY", cspec["dimY"], "dimC", cspec["dimC"], "stride", cspec["stride"], "vth",
                  cspec["compProto"].vThMant)

        # re-order the compartment group to SlayerPyTorch convention before the fully connected layers
        self.lastConvLayer = self.reorderLayer(self.layer2)

        self.convNumInputs = self.lastConvLayer.numNodes

    def createConvLayers2(self, conv_wgt):

        self.isConv = True

        biasEx = 0  # bias exponential
        biasMn = 1  # bias mantissa default
        self.convTh = 0.3

        def get_convcompProto(vTh):
            convProto = nx.CompartmentPrototype(
                # logicalCoreId=coreIdx,
                compartmentCurrentDecay=4095,
                compartmentVoltageDecay=0,
                # biasExp=0,
                biasMant=biasMn,
                biasExp=biasEx,
                vThMant=vTh,  # i.e. 2 * 64 = 128
                refractoryDelay=1,
                vMinExp=20,
                # enableHomeostasis=0,
                # numDendriticAccumulators=64,

                # enableNoise=1,
                # randomizeVoltage=1,
                # noiseMantAtCompartment=5,
                # noiseExpAtCompartment=6,

                functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
                thresholdBehavior=nx.COMPARTMENT_THRESHOLD_MODE.SPIKE_AND_RESET
            )
            return convProto

        def get_poolcompProto(vTh):
            poolProto = nx.CompartmentPrototype(
                # logicalCoreId=coreIdx,
                compartmentCurrentDecay=4095,
                compartmentVoltageDecay=0,
                # biasExp=0,
                biasMant=biasMn,
                biasExp=biasEx,
                vThMant=vTh,  # i.e. 2 * 64 = 128
                refractoryDelay=1,
                vMinExp=20,
                # enableHomeostasis=0,
                # numDendriticAccumulators=64,
                # enableNoise=1,
                # randomizeVoltage=1,
                # noiseMantAtCompartment=5,
                # noiseExpAtCompartment=5,
                functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
                thresholdBehavior=nx.COMPARTMENT_THRESHOLD_MODE.SPIKE_AND_RESET
            )
            return poolProto

        convSpec = dict()
        convSpec["dimX"] = 5
        convSpec["dimY"] = 5
        convSpec["dimC"] = 16
        convSpec["stride"] = 2
        convSpec["weightFile"] = conv_wgt[0]
        wsrc = convSpec["dimX"] * convSpec["dimY"] * 1
        vTh = init_th(wsrc, 1, self.convTh, 255)
        convSpec["compProto"] = get_convcompProto(vTh)
        compartmentsPerCore = 4096 / 8
        self.convSpec.append(convSpec)

        self.layer1, corenum = self.convLayer(self.allMCSomaGrp.soma, convSpec, self.lastUsedLogicalCoreId + 1,
                                              compartmentsPerCore)

        self.lastUsedLogicalCoreId = int(np.ceil(corenum))

        # poolSpec = dict()
        # poolSpec["stride"] = 3
        # poolSpec["compProto"] = poolProto
        # # poolSpec["weightFile"] = np.ones((poolSpec["stride"], poolSpec["stride"]), dtype=int) * 28*4
        # poolSpec["weightFile"] = int(28*(4/poolSpec["stride"]))
        # compartmentsPerCore = 4096 / 16
        #
        # self.layer2, corenum = self.poolingLayer(self.layer1, poolSpec, self.lastUsedLogicalCoreId+1, compartmentsPerCore)
        #
        # self.lastUsedLogicalCoreId = int(np.ceil(corenum))

        convSpec = dict()
        convSpec["dimX"] = 3
        convSpec["dimY"] = 3
        convSpec["dimC"] = 8
        convSpec["stride"] = 2
        convSpec["weightFile"] = conv_wgt[1]
        wsrc = convSpec["dimX"] * convSpec["dimY"] * self.convSpec[0]["dimC"]
        vTh = init_th(wsrc, 1, self.convTh, 255)
        convSpec["compProto"] = get_convcompProto(vTh)
        compartmentsPerCore = 4096 / 8
        self.convSpec.append(convSpec)

        self.layer2, corenum = self.convLayer(self.layer1, convSpec, self.lastUsedLogicalCoreId + 1,
                                              compartmentsPerCore)

        self.lastUsedLogicalCoreId = int(np.ceil(corenum))

        for i, cspec in enumerate(self.convSpec):
            print("conv_layer_", i)
            print("dimX", cspec["dimX"], "dimY", cspec["dimY"], "dimC", cspec["dimC"], "stride", cspec["stride"], "vth",
                  cspec["compProto"].vThMant)

        # re-order the compartment group to SlayerPyTorch convention before the fully connected layers
        self.lastConvLayer = self.reorderLayer(self.layer2)

        self.convNumInputs = self.lastConvLayer.numNodes

    def distributeCompartments(self, layer, corenum, compartmentsPerCore):
        """Distributes compartments across cores, starting on the next available
        core as determined from corenum
        :param CompartmentGroup layer: The group of compartments to distribute
        :param float corenum: The last used logicalCoreId
        :param int compartmentsPerCore: The maximum number of compartments per core
        :returns: The last used logicalCoreId
        :rtype: float
        """

        corenum = np.ceil(corenum)
        for comp in layer:
            comp.logicalCoreId = int(np.floor(corenum))
            corenum = corenum + 1 / compartmentsPerCore

        return corenum

    def reorderLayer(self, layerIn):
        """
        Converts a compartment group from WHC to CHW order.
        :param CompartmentGroup layerIn: The layer to reorder.
        :returns: The re-ordered layer
        :rtype: CompartmentGroup
        """

        net = layerIn.net

        layerOut = net.createCompartmentGroup()
        layerOut.sizeX = layerIn.sizeX
        layerOut.sizeY = layerIn.sizeY
        layerOut.sizeC = layerIn.sizeC

        for cc in range(layerIn.sizeC):
            for yy in range(layerIn.sizeY):
                for xx in range(layerIn.sizeX):
                    layerOut.addCompartments(layerIn[xx * layerIn.sizeY * layerIn.sizeC + yy * layerIn.sizeC + cc])

        return layerOut

    def fullLayer(self, layerInput, fullSpec, corenum, compartmentsPerCore):
        """Create a new fully connected layer.
        :param CompartmentGroup layerInput: The input to the fully connected layer
        :param dict fullSpec: Specifies "dim", the number of neurons,
                             "connProto", "compProto" prototypes of the layer,
                             "weightFile" where the weights can be read in from.
        :param float corenum: The last output of distributeCompartments()
        :param int compartmentsPerCore: The maximum number of compartments per core
        :returns:
            - layerOutput (CompartmentGroup): The compartments of the fully connected layer
            - corenum (float): The last used logicalCoreId
        """
        # properties of the input layer
        nInput = layerInput.numNodes
        net = layerInput.net

        # properties of the convolution function
        compProto = fullSpec["compProto"]
        dim = fullSpec["dim"]
        weight = fullSpec["weightFile"]
        if "delayFile" in fullSpec:
            delayFile = fullSpec["delayFile"]
        else:
            delayFile = None

        if delayFile is not None:
            D = np.load(delayFile)
        else:
            D = np.zeros((nInput,))

        maxD = np.max(D)
        if maxD != 0:
            numDelayBits = np.ceil(np.log2(maxD))
            enableDelay = 1
        else:
            numDelayBits = 0
            enableDelay = 0

        connProto = nx.ConnectionPrototype(signMode=nx.SYNAPSE_SIGN_MODE.MIXED,
                                           numDelayBits=numDelayBits,
                                           enableDelay=enableDelay,
                                           numTagBits=0,
                                           compressionMode=3)
        nOutput = dim

        # weight = np.load(weightFile)
        delay = np.zeros((nOutput, nInput))
        for ii in range(nInput):
            delay[:, ii] = D[ii]

        for ii in [64, 32, 16, 8]:
            if ii > maxD + 1:
                compProto.numDendriticAccumulators = ii

        layerOutput = net.createCompartmentGroup(size=nOutput, prototype=compProto)

        layerInput.connect(layerOutput,
                           prototype=connProto,
                           weight=weight,
                           delay=delay)

        corenum = self.distributeCompartments(layerOutput, corenum, compartmentsPerCore)

        connProto.delay = 0
        connProto.numDelayBits = 0
        connProto.enableDelay = 0
        return layerOutput, corenum

    def poolingLayer(self, layerInput, convSpec, corenum, compartmentsPerCore):
        """Create a new pooling layer.
        :param CompartmentGroup layerInput: The input to the convolution layer
        :param dict convSpec: Specifies "dimX","dimY","dimC" of the filter,
                             "connProto", "compProto" prototypes of the layer,
                             "weightFile" where the weights can be read in from.
        :param float corenum: The last output of distributeCompartments()
        :param int compartmentsPerCore: The maximum number of compartments per core
        :returns:
            - layerOutput (CompartmentGroup): The compartments of the convolution layer
            - corenum (float): The last used logicalCoreId
        """
        # properties of the input layer
        sizeXin = layerInput.sizeX
        sizeYin = layerInput.sizeY
        sizeCin = layerInput.sizeC
        nInput = sizeXin * sizeYin * sizeCin
        net = layerInput.net

        # properties of the convolution function
        # stride = convSpec["stride"] #not implemented yet
        compProto = convSpec["compProto"]
        convX = convSpec["stride"]
        convY = convSpec["stride"]
        # convC = convSpec["dimC"]
        W = convSpec["weightFile"]

        inputShape = [layerInput.sizeY, layerInput.sizeX, layerInput.sizeC]
        kernelShape = [convSpec["stride"], convSpec["stride"]]
        strides = [convSpec["stride"], convSpec["stride"]]
        zeroPadding = None
        isDepthwise = True
        padding = []
        dilation = [1, 1]

        if isDepthwise:
            inputChannels = 1
            inputShift = inputShape[0] * inputShape[1]
        else:
            inputChannels = inputShape[-1]
            inputShift = 0

        # properties of the output layer
        sizeXout = self.conv_output_shape(sizeXin, kernelShape[0], padding, strides[0], dilation[0])
        sizeYout = self.conv_output_shape(sizeYin, kernelShape[1], padding, strides[1], dilation[1])
        # sizeXout = int(np.ceil((layerInput.sizeX - convSpec["dimX"]) / convSpec["stride"]))
        # sizeYout = int(np.ceil((layerInput.sizeY - convSpec["dimY"]) / convSpec["stride"]))
        sizeCout = sizeCin
        nOutput = sizeXout * sizeYout * sizeCout
        outputShape = [sizeYout, sizeXout, sizeCout]

        # weight = W[dx, dy, cSrc, cDst]
        wt = convSpec["weightFile"]
        W = np.ones((convSpec["stride"], convSpec["stride"], inputChannels, sizeCout), dtype=int) * wt
        W = np.transpose(W, (3, 2, 0, 1))

        layerOutput = net.createCompartmentGroup(size=nOutput, prototype=compProto)
        layerOutput.sizeX = sizeXout
        layerOutput.sizeY = sizeYout
        layerOutput.sizeC = sizeCout

        numDelayBits = 0
        enableDelay = 0

        connProto = nx.ConnectionPrototype(signMode=nx.SYNAPSE_SIGN_MODE.MIXED,
                                           numDelayBits=numDelayBits,
                                           enableDelay=enableDelay,
                                           numTagBits=0)

        # Subtract zero padding of previous layer from inputShape. Used if previous layer
        # was ZeroPadding.
        if zeroPadding is not None:
            py0, py1, px0, px1 = zeroPadding
            inputShape = (inputShape[0] - (py0 + py1), inputShape[1] - (px0 + px1), inputShape[2])

        inputSize = np.asscalar(np.prod(inputShape))
        outputSize = np.asscalar(np.prod(outputShape))
        numStrides = np.asscalar(np.prod(outputShape[:-1]))
        outIds = np.arange(numStrides)
        inputIdMap = np.reshape(np.arange(inputSize), inputShape, 'F')

        # Add zero-padding.
        doPad = np.any(padding)
        if doPad:
            py0, py1, px0, px1 = padding
            inputIdMap = np.pad(inputIdMap, ((py0, py1), (px0, px1), (0, 0)),
                                'constant', constant_values=-1)

        dy, dx = dilation

        # Get indices of input neurons where conv kernel will be applied.
        outIdsY, outIdsX, _ = np.unravel_index(outIds, outputShape, 'F')
        inIdsY = outIdsY * strides[0]
        inIdsX = outIdsX * strides[1]

        # Generate a flat dummy kernel. Need to offset by 1 because lil_matrix does
        # not store zeros.
        kernelSize = np.asscalar(np.prod(kernelShape)) * inputChannels
        # kIds = np.arange(kernelSize) + 1
        kIds = np.arange(kernelSize)

        inputIds = []
        kernelIds = []
        outputIds = []
        for outId, inIdY, inIdX in zip(outIds, inIdsY, inIdsX):
            inIds = inputIdMap[slice(inIdY, inIdY + dy * kernelShape[0], dy),
                               slice(inIdX, inIdX + dx * kernelShape[1], dx),
                               slice(inputChannels)]
            inIds = np.ravel(inIds, 'F')

            # Remove zero-padding.
            if doPad:
                paddingMask = inIds > -1
                inIds = inIds[paddingMask]
                _kIds = kIds[paddingMask]
            else:
                _kIds = kIds

            inputIds.append(inIds)
            kernelIds.append(_kIds)
            outputIds.append([outId] * len(_kIds))

        inputIds = np.concatenate(inputIds)
        kernelIds = np.concatenate(kernelIds)
        outputIds = np.concatenate(outputIds)

        ww = np.reshape(W, newshape=(outputShape[-1], kernelSize))

        # Insert kernel into all channels of feature map.
        data = []
        rows = []
        cols = []
        weights = []
        for cId in range(outputShape[-1]):
            # Increment kernel ids by the kernelSize for the next channel.
            data.append(np.ones(len(kernelIds), dtype=int))
            # data.append(kernelIds + cId * kernelSize)
            cols.append(inputIds + cId * inputShift)
            rows.append(outputIds + cId * numStrides)
            tw = W[cId].ravel()
            weights.append(list(tw[kernelIds]))

        data = np.concatenate(data)
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        weights = np.concatenate(weights)

        kernelIdMap = sps.coo_matrix((data, (rows, cols)), (outputSize, inputSize), int)

        sparseWeights = sps.coo_matrix((weights, (rows, cols)), (outputSize, inputSize), int)

        # aa = sparseWeights.todense()
        # ab = np.squeeze(np.asarray(aa))
        # import matplotlib.pyplot as plt
        # plt.imshow(ab)
        # plt.show()

        layerInput.connect(layerOutput,
                           prototype=connProto,
                           connectionMask=kernelIdMap,
                           weight=sparseWeights)

        corenum = self.distributeCompartments(layerOutput, corenum, compartmentsPerCore)
        # connProto.delay = 0
        connProto.numDelayBits = 0
        connProto.enableDelay = 0
        return layerOutput, corenum

    def conv_output_shape(self, input_length, filter_size, padding, stride, dilation=1):
        """Determines output length of a convolution given input length.
        Arguments:
            input_length: integer.
            filter_size: integer.
            padding: one of "same", "valid", "full", "causal"
            stride: integer.
            dilation: dilation rate, integer.
        Returns:
            The output length (integer).
        """

        dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
        output_length = input_length - dilated_filter_size + 1
        final_output_length = (output_length + stride - 1) // stride

        return final_output_length

    def convLayer(self, layerInput, convSpec, corenum, compartmentsPerCore):
        """Generate a KernelIdMap of the layer.
        :param CompartmentGroup layerInput: The input to the convolution layer
        :param dict convSpec: Specifies "dimX","dimY","dimC" of the filter,
                             "connProto", "compProto" prototypes of the layer,
                             "weightFile" where the weights can be read in from.
        :param float corenum: The last output of distributeCompartments()
        :param int compartmentsPerCore: The maximum number of compartments per core
        :returns:
            - layerOutput (CompartmentGroup): The compartments of the convolution layer
            - corenum (float): The last used logicalCoreId
        """

        # properties of the input layer
        sizeXin = layerInput.sizeX
        sizeYin = layerInput.sizeY
        sizeCin = layerInput.sizeC
        nInput = sizeXin * sizeYin * sizeCin
        net = layerInput.net

        # properties of the convolution function
        # stride = convSpec["stride"] #not implemented yet
        compProto = convSpec["compProto"]
        convX = convSpec["dimX"]
        convY = convSpec["dimY"]
        convC = convSpec["dimC"]
        W = convSpec["weightFile"]

        inputShape = [layerInput.sizeY, layerInput.sizeX, layerInput.sizeC]
        kernelShape = [convSpec["dimY"], convSpec["dimX"]]
        strides = [convSpec["stride"], convSpec["stride"]]
        zeroPadding = None
        isDepthwise = False
        padding = []
        dilation = [1, 1]

        # properties of the output layer
        sizeXout = self.conv_output_shape(sizeXin, kernelShape[0], padding, strides[0], dilation[0])
        sizeYout = self.conv_output_shape(sizeYin, kernelShape[1], padding, strides[1], dilation[1])
        # sizeXout = int(np.ceil((layerInput.sizeX - convSpec["dimX"]) / convSpec["stride"]))
        # sizeYout = int(np.ceil((layerInput.sizeY - convSpec["dimY"]) / convSpec["stride"]))
        sizeCout = convSpec["dimC"]
        nOutput = sizeXout * sizeYout * sizeCout
        outputShape = [sizeYout, sizeXout, sizeCout]

        # weight = W[dx, dy, cSrc, cDst]
        W = convSpec["weightFile"]
        W = np.transpose(W, (3, 2, 0, 1))
        # W = np.transpose(W, (3, 2, 1, 0))

        layerOutput = net.createCompartmentGroup(size=nOutput, prototype=compProto)
        layerOutput.sizeX = sizeXout
        layerOutput.sizeY = sizeYout
        layerOutput.sizeC = sizeCout

        numDelayBits = 0
        enableDelay = 0

        connProto = nx.ConnectionPrototype(signMode=nx.SYNAPSE_SIGN_MODE.MIXED,
                                           numDelayBits=numDelayBits,
                                           enableDelay=enableDelay,
                                           numTagBits=0)

        # Subtract zero padding of previous layer from inputShape. Used if previous layer
        # was ZeroPadding.
        if zeroPadding is not None:
            py0, py1, px0, px1 = zeroPadding
            inputShape = (inputShape[0] - (py0 + py1), inputShape[1] - (px0 + px1), inputShape[2])

        inputSize = np.asscalar(np.prod(inputShape))
        outputSize = np.asscalar(np.prod(outputShape))
        numStrides = np.asscalar(np.prod(outputShape[:-1]))
        outIds = np.arange(numStrides)
        inputIdMap = np.reshape(np.arange(inputSize), inputShape, 'F')
        if isDepthwise:
            inputChannels = 1
            inputShift = inputShape[0] * inputShape[1]
        else:
            inputChannels = inputShape[-1]
            inputShift = 0

        # Add zero-padding.
        doPad = np.any(padding)
        if doPad:
            py0, py1, px0, px1 = padding
            inputIdMap = np.pad(inputIdMap, ((py0, py1), (px0, px1), (0, 0)),
                                'constant', constant_values=-1)

        dy, dx = dilation

        # Get indices of input neurons where conv kernel will be applied.
        outIdsY, outIdsX, _ = np.unravel_index(outIds, outputShape, 'F')
        inIdsY = outIdsY * strides[0]
        inIdsX = outIdsX * strides[1]

        # Generate a flat dummy kernel. Need to offset by 1 because lil_matrix does
        # not store zeros.
        kernelSize = np.asscalar(np.prod(kernelShape)) * inputChannels
        # kIds = np.arange(kernelSize) + 1
        kIds = np.arange(kernelSize)

        inputIds = []
        kernelIds = []
        outputIds = []
        for outId, inIdY, inIdX in zip(outIds, inIdsY, inIdsX):
            inIds = inputIdMap[slice(inIdY, inIdY + dy * kernelShape[0], dy),
                               slice(inIdX, inIdX + dx * kernelShape[1], dx),
                               slice(inputChannels)]
            inIds = np.ravel(inIds, 'F')

            # Remove zero-padding.
            if doPad:
                paddingMask = inIds > -1
                inIds = inIds[paddingMask]
                _kIds = kIds[paddingMask]
            else:
                _kIds = kIds

            inputIds.append(inIds)
            kernelIds.append(_kIds)
            outputIds.append([outId] * len(_kIds))

        inputIds = np.concatenate(inputIds)
        kernelIds = np.concatenate(kernelIds)
        outputIds = np.concatenate(outputIds)

        ww = np.reshape(W, newshape=(outputShape[-1], kernelSize))

        # Insert kernel into all channels of feature map.
        data = []
        rows = []
        cols = []
        weights = []
        for cId in range(outputShape[-1]):
            # Increment kernel ids by the kernelSize for the next channel.
            data.append(np.ones(len(kernelIds), dtype=int))
            # data.append(kernelIds + cId * kernelSize)
            cols.append(inputIds + cId * inputShift)
            rows.append(outputIds + cId * numStrides)
            tw = W[cId].ravel()
            weights.append(list(tw[kernelIds]))

        data = np.concatenate(data)
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        weights = np.concatenate(weights)

        kernelIdMap = sps.coo_matrix((data, (rows, cols)), (outputSize, inputSize), int)

        sparseWeights = sps.coo_matrix((weights, (rows, cols)), (outputSize, inputSize), int)

        # aa = sparseWeights.todense()
        # ab = np.squeeze(np.asarray(aa))
        # import matplotlib.pyplot as plt
        # plt.imshow(ab)
        # plt.show()

        layerInput.connect(layerOutput,
                           prototype=connProto,
                           connectionMask=kernelIdMap,
                           weight=sparseWeights)

        corenum = self.distributeCompartments(layerOutput, corenum, compartmentsPerCore)
        # connProto.delay = 0
        connProto.numDelayBits = 0
        connProto.enableDelay = 0
        return layerOutput, corenum

    def createGCNeuronsPerPattern(self, patternIdx):
        """ configures the GC neurons for each pattern"""
        # max number of neurons in a core
        if patternIdx == self.numlayers - 1:
            maxColsPerCore = self.numTargets
            maxColsPerCore = 5
            vminexp = 0
        else:
            maxColsPerCore = 10
            vminexp = 0

        # the following parameters should get ~87% for MNIST
        ####################
        self.inhid_vth = 0.5  # first layer threshold
        self.hid_vth = 0.5  # middle layer threshold
        self.classifier_vth = 0.3  # classifier layer threshold
        self.biasEx = 0  # bias exponential
        self.biasMn = 1  # bias mantissa default

        scale = 1
        self.GCtoECDelayDeriv = int(62)
        # self.ECtoGCDelayDeriv = int(2)
        self.wtadelay = int(0)
        # self.lastECdelay = int(0)
        self.voldcy = int(0)
        self.curdcy = int(4095)

        if self.trainbool != 1:
            self.wtadelay = int(0)
            self.voldcy = int(0)

        self.ECtoGCwgt = 255
        self.LabeltoECwgt = 8

        thold = 0
        wsrc = 0

        tmps = 0

        # calculating forward and error path thresholds
        if patternIdx == self.numlayers - 1:
            wsrc = self.numHidNurns[patternIdx - 1]
            thold = init_th(wsrc, patternIdx, self.classifier_vth, 255)
            self.biasMn = int(thold / 10)
            ref = 1
            ethold = int(self.LabeltoECwgt)
            # ethold = int(1 * init_th(10, patternIdx, self.classifier_vth, self.bposwgtrng) / (
            #         0 + 1))
            print("ethreshold = ")
            print(ethold)
        elif patternIdx == 0:
            if self.convNumInputs == 0:
                wsrc = self.numMCs
            else:
                wsrc = self.convNumInputs
            thold = init_th(wsrc, patternIdx, self.inhid_vth, 255)
            self.biasMn = int(thold / 10)
            ref = 1
            if self.numlayers == 2:
                ethold = int(self.LabeltoECwgt)
            else:
                ethold = int(
                    1 * init_th(self.numHidNurns[patternIdx + 1], patternIdx, self.inhid_vth, self.bposwgtrng) / (
                                patternIdx + 2))
                # ethold = int(self.LabeltoECwgt) + (self.numlayers - patternIdx - 2)*ethold
            # ethold = int(self.LabeltoECwgt) * self.numHidNurns[patternIdx + 1]
            # ethold = int(1 * init_th(self.numHidNurns[patternIdx + 1], patternIdx, self.inhid_vth, self.bposwgtrng) / (patternIdx + 1))
            # ethold = int(self.LabeltoECwgt)*self.numHidNurns[patternIdx + 1]
            # ethold = int(self.LabeltoECwgt * (self.numHidNurns[patternIdx + 1] / 2.0))
            # ethold = int(self.LabeltoECwgt)
            # ethold = int(1 * init_th(self.numHidNurns[patternIdx + 1], patternIdx, self.inhid_vth, self.bposwgtrng) / (
            #             0 + 1))
            print("ethreshold = ")
            print(ethold)
        else:
            wsrc = self.numHidNurns[patternIdx - 1]
            thold = init_th(wsrc, patternIdx, self.hid_vth, 255)
            self.biasMn = int(thold / 10)
            ref = 1
            # ethold = int(1*init_th(self.numHidNurns[patternIdx + 1], patternIdx, self.hid_vth, self.bposwgtrng) / (patternIdx + 1))
            # # ethold = int(self.LabeltoECwgt) + (self.numlayers - patternIdx - 2)*ethold
            # ethold = int(self.LabeltoECwgt) * self.numHidNurns[patternIdx + 1]
            ethold = int(self.LabeltoECwgt * (self.numHidNurns[patternIdx + 1] / 2.0))
            # ethold = int(self.LabeltoECwgt)
            # ethold = int(1 * init_th(self.numHidNurns[patternIdx + 1], patternIdx, self.hid_vth, self.bposwgtrng) / (
            #             0 + 1))
            print("ethreshold = ")
            print(ethold)

        # ethold = int(thold / 200)
        # ethold = int(self.LabeltoECwgt)
        # ethold = 32

        numTmps = int(np.ceil(thold / 255))
        numTmpsNeurons = 1

        print("threshold = ")
        print(thold)

        # mapping neurons of a layer into cores one at a time
        for colIdx in range(self.numHidNurns[patternIdx]):
            #             coreIdx = self.lastUsedLogicalCoreId + math.floor((colIdx+1)/maxColsPerCore)*2
            coreIdx = self.lastUsedLogicalCoreId + math.floor((colIdx) / maxColsPerCore) * 2

            # compartment prototype for the main forward path neurons
            gcProto = nx.CompartmentPrototype(
                logicalCoreId=coreIdx + 1,
                compartmentCurrentDecay=self.curdcy,
                compartmentVoltageDecay=self.voldcy,
                enableSpikeBackprop=1,
                enableSpikeBackpropFromSelf=1,
                biasMant=self.biasMn,
                biasExp=self.biasEx,
                vThMant=thold,
                vMinExp=20,
                refractoryDelay=ref,
                # numDendriticAccumulators=64,

                # activityTimeConstant=64,
                # activityImpulse=1,
                # minActivity=40,
                # maxActivity=80,
                # homeostasisGain=1,

                # enableNoise=1,
                # # randomizeVoltage=1,
                # noiseMantAtCompartment=0,
                # noiseExpAtCompartment=10,

                functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
                thresholdBehavior=nx.COMPARTMENT_THRESHOLD_MODE.SPIKE_AND_RESET,
            )

            gcCx = self.net.createCompartment(prototype=gcProto)
            self.allGCsPerPattern[patternIdx].addCompartments(gcCx)

            if patternIdx == self.numlayers - 1:
                tmps = colIdx
            else:
                tmps = 0
            # compartment prototype for the error path neurons, +ve error neurons
            ecProtoPos = nx.CompartmentPrototype(
                logicalCoreId=coreIdx + 2,
                compartmentCurrentDecay=4095,
                compartmentVoltageDecay=0+tmps,
                # enableSpikeBackprop=1,
                # enableSpikeBackpropFromSelf=1,
                biasMant=0,
                vThMant=ethold+tmps,
                vMinExp=vminexp,
                refractoryDelay=1,
                numDendriticAccumulators=64,
                functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
                thresholdBehavior=nx.COMPARTMENT_THRESHOLD_MODE.SPIKE_AND_RESET,
            )

            # compartment prototype for the error path neurons, -ve error neurons
            ecProtoNeg = nx.CompartmentPrototype(
                logicalCoreId=coreIdx + 2,
                compartmentCurrentDecay=4095,
                compartmentVoltageDecay=0+tmps,
                # enableSpikeBackprop=1,
                # enableSpikeBackpropFromSelf=1,
                biasMant=0,
                vThMant=ethold+tmps,
                vMinExp=vminexp,
                refractoryDelay=1,
                numDendriticAccumulators=64,
                functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
                thresholdBehavior=nx.COMPARTMENT_THRESHOLD_MODE.SPIKE_AND_RESET,
            )

            # compartment prototype for intermediate neurons from error to forward path, +ve intermediate error neurons
            ecTmpPos = nx.CompartmentPrototype(
                logicalCoreId=coreIdx + 2,
                compartmentCurrentDecay=4095,
                compartmentVoltageDecay=0,
                # enableSpikeBackprop=1,
                # enableSpikeBackpropFromSelf=1,
                biasMant=0,
                vThMant=2,
                vMinExp=vminexp,
                refractoryDelay=1,
                numDendriticAccumulators=64,
                functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
                thresholdBehavior=nx.COMPARTMENT_THRESHOLD_MODE.SPIKE_AND_RESET,
            )

            # compartment prototype for intermediate neurons from error to forward path, -ve intermediate error neurons
            ecTmpNeg = nx.CompartmentPrototype(
                logicalCoreId=coreIdx + 2,
                compartmentCurrentDecay=4095,
                compartmentVoltageDecay=0,
                # enableSpikeBackprop=1,
                # enableSpikeBackpropFromSelf=1,
                biasMant=0,
                vThMant=2,
                vMinExp=vminexp,
                refractoryDelay=1,
                numDendriticAccumulators=64,
                functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
                thresholdBehavior=nx.COMPARTMENT_THRESHOLD_MODE.SPIKE_AND_RESET,
            )

            if patternIdx != self.numlayers - 1:

                # compartment prototype for auxilary compartment in error path for derivative, +ve derivative error neurons
                ecProtoPosAux = nx.CompartmentPrototype(
                    logicalCoreId=coreIdx + 2,
                    compartmentCurrentDecay=0,
                    compartmentVoltageDecay=0,
                    # enableSpikeBackprop=1,
                    # enableSpikeBackpropFromSelf=1,
                    biasMant=0,
                    vThMant=2,
                    vMinExp=vminexp,
                    numDendriticAccumulators=64,
                    functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
                    thresholdBehavior=nx.COMPARTMENT_THRESHOLD_MODE.SPIKE_AND_RESET,
                )
                # compartment prototype for auxilary compartment in error path for derivative, -ve derivative error neurons
                ecProtoNegAux = nx.CompartmentPrototype(
                    logicalCoreId=coreIdx + 2,
                    compartmentCurrentDecay=0,
                    compartmentVoltageDecay=0,
                    # enableSpikeBackprop=1,
                    # enableSpikeBackpropFromSelf=1,
                    biasMant=0,
                    vThMant=2,
                    vMinExp=vminexp,
                    numDendriticAccumulators=64,
                    functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
                    thresholdBehavior=nx.COMPARTMENT_THRESHOLD_MODE.SPIKE_AND_RESET,
                )

                # derivative operation (i.e. elementwise multiplication) through AND operation between auxilary and soma spikes
                ecProtoPos.addDendrite(ecProtoPosAux, nx.COMPARTMENT_JOIN_OPERATION.PASS)
                ecNeuProtoPos = nx.NeuronPrototype(ecProtoPos)
                ecCxPos = self.net.createNeuron(ecNeuProtoPos)
                self.allPosECsPerPattern[patternIdx].addNeuron(ecCxPos)

                ecProtoNeg.addDendrite(ecProtoNegAux, nx.COMPARTMENT_JOIN_OPERATION.PASS)
                ecNeuProtoNeg = nx.NeuronPrototype(ecProtoNeg)
                ecCxNeg = self.net.createNeuron(ecNeuProtoNeg)
                self.allNegECsPerPattern[patternIdx].addNeuron(ecCxNeg)

                for k in range(numTmpsNeurons):
                    ecCxPos = self.net.createCompartment(prototype=ecTmpPos)
                    self.allTmpPosECsPerPattern[patternIdx].addCompartments(ecCxPos)
                    ecCxNeg = self.net.createCompartment(prototype=ecTmpNeg)
                    self.allTmpNegECsPerPattern[patternIdx].addCompartments(ecCxNeg)
            else:
                ecCxPos = self.net.createCompartment(prototype=ecProtoPos)
                self.allPosECsPerPattern[patternIdx].addCompartments(ecCxPos)
                ecCxNeg = self.net.createCompartment(prototype=ecProtoNeg)
                self.allNegECsPerPattern[patternIdx].addCompartments(ecCxNeg)

                for k in range(numTmpsNeurons):
                    ecCxPos = self.net.createCompartment(prototype=ecTmpPos)
                    self.allTmpPosECsPerPattern[patternIdx].addCompartments(ecCxPos)
                    ecCxNeg = self.net.createCompartment(prototype=ecTmpNeg)
                    self.allTmpNegECsPerPattern[patternIdx].addCompartments(ecCxNeg)

            # if patternIdx == self.numlayers - 1:
            #     labelProto = nx.CompartmentPrototype(
            #         logicalCoreId=coreIdx + 3,
            #         compartmentCurrentDecay=4095,
            #         compartmentVoltageDecay=0,
            #         # enableSpikeBackprop=1,
            #         # enableSpikeBackpropFromSelf=1,
            #         vThMant=2,  # i.e. 2 * 64 = 128
            #         # refractoryDelay=19,
            #         vMinExp=0,
            #         refractoryDelay=1,
            #         numDendriticAccumulators=32,
            #         functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
            #         thresholdBehavior=nx.COMPARTMENT_THRESHOLD_MODE.SPIKE_AND_RESET
            #     )
            #     labelgc = self.net.createCompartment(prototype=labelProto)
            #     self.allLabelGrp.addCompartments(labelgc)
            #
            #     # if self.wtadelay != 0:
            #     # wta network set up but not used based on wtadelay=0
            #     wtaProto = nx.CompartmentPrototype(
            #         logicalCoreId=coreIdx + 3,
            #         compartmentCurrentDecay=0,
            #         compartmentVoltageDecay=0,
            #         # enableSpikeBackprop=1,
            #         # enableSpikeBackpropFromSelf=1,
            #         vThMant=2,  # i.e. 2 * 64 = 128
            #         # refractoryDelay=19,
            #         vMinExp=0,
            #         refractoryDelay=1,
            #         numDendriticAccumulators=32,
            #         functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
            #         thresholdBehavior=nx.COMPARTMENT_THRESHOLD_MODE.SPIKE_AND_RESET
            #     )
            #     wtagc = self.net.createCompartment(prototype=wtaProto)
            #     self.wtaGrp.addCompartments(wtagc)

        if patternIdx == self.numlayers - 1:
            for i in range(self.numHidNurns[patternIdx]):
                tauU = 2
                tauV = 25
                decayU = int(1 / tauU * 2 ** 12)
                decayV = int(1 / tauV * 2 ** 12)
                # decayU = 4095
                # decayV = 0
                vth = 256
                inWgt = 35  # Input spike connection weight

                # create a neuron prototype np0
                np1 = nx.NeuronPrototypes.NeuronSoftResetTwoCompartments(decayU, decayV, vth, logicalCoreId=coreIdx + 3)

                # create a two-compartment neuron with prototype np0
                neuron1 = self.net.createNeuron(np1)

                # Connect somatic compartment with dendritic compartment
                wgtInh = -int(vth * decayU / 4096)
                connProto1 = nx.ConnectionPrototype(weight=wgtInh)
                neuron1.soma.connect(neuron1.dendrites[0], connProto1)
                neuron1.dendrites[0].biasExp = 6
                self.allLabelGrp.addNeuron(neuron1)
                # self.allLabelGrp.soma.dendrites[0]

                # if self.wtadelay != 0:
                # wta network set up but not used based on wtadelay=0
                wtaProto = nx.CompartmentPrototype(
                    logicalCoreId=coreIdx + 3,
                    compartmentCurrentDecay=0,
                    compartmentVoltageDecay=0,
                    # enableSpikeBackprop=1,
                    # enableSpikeBackpropFromSelf=1,
                    vThMant=2,  # i.e. 2 * 64 = 128
                    # refractoryDelay=19,
                    # vMinExp=0,
                    # refractoryDelay=1,
                    # numDendriticAccumulators=32,
                    functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
                    thresholdBehavior=nx.COMPARTMENT_THRESHOLD_MODE.SPIKE_AND_RESET
                )
                wtagc = self.net.createCompartment(prototype=wtaProto)
                self.wtaGrp.addCompartments(wtagc)

        self.lastUsedLogicalCoreId += math.ceil(self.numHidNurns[patternIdx] / maxColsPerCore) * 2

        ij = numTmpsNeurons
        E = sps.identity(self.numHidNurns[patternIdx])  # Sparse identity matrix
        oneVector = np.ones((1, ij))  # Dense
        mmm = sps.kron(E, oneVector)  # m is sparse.
        connmat = mmm.A
        connmat.astype(int)

        # weight exponential from intermediate error neurons to forward path
        ijexp = int(np.ceil(np.log2(numTmps)) + 1)

        # creating connections from the error network to the forward path
        if patternIdx != self.numlayers - 1:
            posECtoTmpECConnProtoBox = nx.ConnectionPrototype(
                weight=10,
                numWeightBits=8,
                signMode=nx.SYNAPSE_SIGN_MODE.MIXED,
            )

            negECtoTmpECConnProtoBox = nx.ConnectionPrototype(
                weight=10,
                numWeightBits=8,
                signMode=nx.SYNAPSE_SIGN_MODE.MIXED,
            )

            posECtoTmpECConns = self.net.createConnectionGroup(
                src=self.allPosECsPerPattern[patternIdx].soma,
                dst=self.allTmpPosECsPerPattern[patternIdx],
                prototype=posECtoTmpECConnProtoBox,
                connectionMask=connmat.T
            )

            negECtoTmpECConns = self.net.createConnectionGroup(
                src=self.allNegECsPerPattern[patternIdx].soma,
                dst=self.allTmpNegECsPerPattern[patternIdx],
                prototype=negECtoTmpECConnProtoBox,
                connectionMask=connmat.T
            )

            tmpposECtoGCConnProtoBox = nx.ConnectionPrototype(
                weight=self.ECtoGCwgt,
                numWeightBits=8,
                weightExponent=ijexp,
                signMode=nx.SYNAPSE_SIGN_MODE.MIXED,
            )

            tmpnegECtoGCConnProtoBox = nx.ConnectionPrototype(
                weight=-self.ECtoGCwgt,
                numWeightBits=8,
                weightExponent=ijexp,
                signMode=nx.SYNAPSE_SIGN_MODE.MIXED,
            )

            tmpposECtoGCConns = self.net.createConnectionGroup(
                src=self.allTmpPosECsPerPattern[patternIdx],
                dst=self.allGCsPerPattern[patternIdx],
                prototype=tmpposECtoGCConnProtoBox,
                connectionMask=connmat
            )

            tmpnegECtoGCConns = self.net.createConnectionGroup(
                src=self.allTmpNegECsPerPattern[patternIdx],
                dst=self.allGCsPerPattern[patternIdx],
                prototype=tmpnegECtoGCConnProtoBox,
                connectionMask=connmat
            )

            ##########################
            # creating connections from the forward path to auxilary error compartments to perform the derivative
            GCtoECConnProtoBox = nx.ConnectionPrototype(
                weight=10,
                numWeightBits=8,
                delay=self.GCtoECDelayDeriv,
                numDelayBits=6,
                enableDelay=1,
                signMode=nx.SYNAPSE_SIGN_MODE.MIXED,
                postSynResponseMode=nx.SYNAPSE_POST_SYN_RESPONSE_MODE.BOX,
            )

            posGCtoECConns = self.net.createConnectionGroup(
                src=self.allGCsPerPattern[patternIdx],
                dst=self.allPosECsPerPattern[patternIdx].dendrites[0],
                prototype=GCtoECConnProtoBox,
                connectionMask=np.eye(self.numHidNurns[patternIdx])
            )

            negGCtoECConns = self.net.createConnectionGroup(
                src=self.allGCsPerPattern[patternIdx],
                dst=self.allNegECsPerPattern[patternIdx].dendrites[0],
                prototype=GCtoECConnProtoBox,
                connectionMask=np.eye(self.numHidNurns[patternIdx])
            )

        # for classifier layer
        if patternIdx == self.numlayers - 1:
            ################################### wta conns
            # wta set up (not used currently)
            if self.wtadelay != 0:
                GCtoWtaConnProtoBox = nx.ConnectionPrototype(
                    weight=self.LabeltoECwgt,
                    numWeightBits=8,
                    delay=self.wtadelay,
                    numDelayBits=5,
                    enableDelay=1,
                    signMode=nx.SYNAPSE_SIGN_MODE.MIXED,
                    postSynResponseMode=nx.SYNAPSE_POST_SYN_RESPONSE_MODE.BOX,
                )

                GCtoWtaConn = self.net.createConnectionGroup(
                    src=self.allGCsPerPattern[patternIdx],
                    dst=self.wtaGrp,
                    prototype=GCtoWtaConnProtoBox,
                    connectionMask=np.eye(self.numHidNurns[patternIdx])
                )

                WtatoGCConnProtoBox = nx.ConnectionPrototype(
                    weight=-200,
                    numWeightBits=8,
                    # weightExponent=ijexp,
                    signMode=nx.SYNAPSE_SIGN_MODE.MIXED,
                )

                WtatoGCConn = self.net.createConnectionGroup(
                    src=self.wtaGrp,
                    dst=self.allGCsPerPattern[patternIdx],
                    prototype=WtatoGCConnProtoBox,
                    connectionMask=np.ones(
                        (self.numHidNurns[patternIdx], self.numHidNurns[patternIdx])) - np.eye(
                        self.numHidNurns[patternIdx])
                )

            ########################################################
            # loss computation through spikes at the top layer of error path using connections from classifier and label
            labelscale = 1.5
            gcscale = 2

            # labelscale = 1
            # gcscale = 1

            GCtoPosECConnProtoBox = nx.ConnectionPrototype(
                weight=-int(self.LabeltoECwgt * gcscale),
                numWeightBits=8,
                signMode=nx.SYNAPSE_SIGN_MODE.MIXED,
            )

            GCtoPosECConn = self.net.createConnectionGroup(
                src=self.allGCsPerPattern[patternIdx],
                dst=self.allPosECsPerPattern[patternIdx],
                prototype=GCtoPosECConnProtoBox,
                connectionMask=np.eye(self.numHidNurns[patternIdx]),
                weight=np.array(-int(self.LabeltoECwgt * gcscale))
            )

            LabeltoPosECConnProtoBox = nx.ConnectionPrototype(
                weight=int(self.LabeltoECwgt * labelscale),
                numWeightBits=8,
                signMode=nx.SYNAPSE_SIGN_MODE.MIXED,
            )

            LabeltoPosECConn = self.net.createConnectionGroup(
                src=self.allLabelGrp.soma,
                dst=self.allPosECsPerPattern[patternIdx],
                prototype=LabeltoPosECConnProtoBox,
                connectionMask=np.eye(self.numHidNurns[patternIdx]),
                weight=np.array(int(self.LabeltoECwgt * labelscale))
            )

            GCtoNegECConnProtoBox = nx.ConnectionPrototype(
                weight=int(self.LabeltoECwgt * labelscale),
                numWeightBits=8,
                signMode=nx.SYNAPSE_SIGN_MODE.MIXED,
            )

            GCtoNegECConn = self.net.createConnectionGroup(
                src=self.allGCsPerPattern[patternIdx],
                dst=self.allNegECsPerPattern[patternIdx],
                prototype=GCtoNegECConnProtoBox,
                connectionMask=np.eye(self.numHidNurns[patternIdx]),
                weight=np.array(int(self.LabeltoECwgt * labelscale))
            )

            LabeltoNegECConnProtoBox = nx.ConnectionPrototype(
                weight=-int(self.LabeltoECwgt * gcscale),
                numWeightBits=8,
                signMode=nx.SYNAPSE_SIGN_MODE.MIXED,
            )

            LabeltoNegECConn = self.net.createConnectionGroup(
                src=self.allLabelGrp.soma,
                dst=self.allNegECsPerPattern[patternIdx],
                prototype=LabeltoNegECConnProtoBox,
                connectionMask=np.eye(self.numHidNurns[patternIdx]),
                weight=np.array(-int(self.LabeltoECwgt * gcscale))
            )

            #################
            # connections from error path to classifier compartments
            posECtoTmpECConnProtoBox = nx.ConnectionPrototype(
                weight=10,
                numWeightBits=8,
                signMode=nx.SYNAPSE_SIGN_MODE.MIXED,
            )

            negECtoTmpECConnProtoBox = nx.ConnectionPrototype(
                weight=10,
                numWeightBits=8,
                signMode=nx.SYNAPSE_SIGN_MODE.MIXED,
            )

            posECtoTmpECConns = self.net.createConnectionGroup(
                src=self.allPosECsPerPattern[patternIdx],
                dst=self.allTmpPosECsPerPattern[patternIdx],
                prototype=posECtoTmpECConnProtoBox,
                connectionMask=connmat.T
            )

            negECtoTmpECConns = self.net.createConnectionGroup(
                src=self.allNegECsPerPattern[patternIdx],
                dst=self.allTmpNegECsPerPattern[patternIdx],
                prototype=negECtoTmpECConnProtoBox,
                connectionMask=connmat.T
            )

            tmpposECtoGCConnProtoBox = nx.ConnectionPrototype(
                weight=int(self.ECtoGCwgt / 1),
                numWeightBits=8,
                weightExponent=ijexp,
                signMode=nx.SYNAPSE_SIGN_MODE.MIXED,
            )

            tmpnegECtoGCConnProtoBox = nx.ConnectionPrototype(
                weight=-int(self.ECtoGCwgt / 1),
                numWeightBits=8,
                weightExponent=ijexp,
                signMode=nx.SYNAPSE_SIGN_MODE.MIXED,
            )

            tmpposECtoGCConns = self.net.createConnectionGroup(
                src=self.allTmpPosECsPerPattern[patternIdx],
                dst=self.allGCsPerPattern[patternIdx],
                prototype=tmpposECtoGCConnProtoBox,
                connectionMask=connmat
            )

            tmpnegECtoGCConns = self.net.createConnectionGroup(
                src=self.allTmpNegECsPerPattern[patternIdx],
                dst=self.allGCsPerPattern[patternIdx],
                prototype=tmpnegECtoGCConnProtoBox,
                connectionMask=connmat
            )

    def connectforwardConns(self, train, wgt):
        """ creates the GC->MC inhibitory connections for each pattern"""
        ConnGroup = namedtuple("ConnGroup", "positive negative")

        # updating weights at the end of the second phase (learning rule modified as sum of products)
        # d -> pre-synaptic spike count (currently not used)
        # x1 -> decaying pre-synaptic spike count
        # y1 -> post-synaptic spike count
        # t -> post-synaptic spike count over the two phases

        lr = 4  # 2^-lr learning rate
        lrt = lr + 1  # top layer learning rate
        lp = 7  # u7 -> 2^7 learning period

        top_x1TimeConstant = 64
        top_y1TimeConstant = 64

        hid_x1TimeConstant = 64
        hid_y1TimeConstant = 64

        dw_top = '2^-' + str(lrt) + '*u' + str(lp) + '*y1*x1 - 2^-' + str(lrt + 1) + '*u' + str(lp) + '*t*x1'
        dw_hid = '2^-' + str(lr) + '*u' + str(lp) + '*y1*x1 - 2^-' + str(lr + 1) + '*u' + str(lp) + '*t*x1'

        for pIdx in range(self.numlayers):
            if pIdx == self.numlayers - 1:
                # single update per sample
                lrule = self.net.createLearningRule(
                    # dd='2^0*x0 - 2^3*u7*d',
                    dt='2^0*y0 - 2^0*u7*t',
                    dw=dw_top,  # add decay term
                    x1Impulse=1,
                    x1TimeConstant=top_x1TimeConstant,
                    y1Impulse=1,
                    y1TimeConstant=top_y1TimeConstant,
                    x2Impulse=1,
                    x2TimeConstant=64,
                    tEpoch=1)
            elif pIdx == 0:
                # single update per sample
                lrule = self.net.createLearningRule(
                    # dd='2^0*x0 - 2^3*u7*d',
                    dt='2^0*y0 - 2^0*u7*t',
                    dw=dw_hid,
                    x1Impulse=1,
                    x1TimeConstant=hid_x1TimeConstant,
                    y1Impulse=1,
                    y1TimeConstant=hid_y1TimeConstant,
                    x2Impulse=1,
                    x2TimeConstant=64,
                    tEpoch=1)
            else:
                # single update per sample
                lrule = self.net.createLearningRule(
                    # dd='2^0*x0 - 2^3*u7*d',
                    dt='2^0*y0 - 2^0*u7*t',
                    dw=dw_hid,
                    x1Impulse=1,
                    x1TimeConstant=hid_x1TimeConstant,
                    y1Impulse=1,
                    y1TimeConstant=hid_y1TimeConstant,
                    x2Impulse=1,
                    x2TimeConstant=64,
                    tEpoch=1)

            # connections for the forward path
            if train:
                # single update per phase
                forwardConnProtoBox = nx.ConnectionPrototype(
                    # weight=4,
                    enableLearning=1,
                    enableDelay=False,
                    numDelayBits=0,
                    learningRule=lrule, numTagBits=8, numWeightBits=8,
                    # weightLimitMant=10, weigthLimitExp=4,
                    signMode=nx.SYNAPSE_SIGN_MODE.MIXED)
            else:
                forwardConnProtoBox = nx.ConnectionPrototype(
                    # weight=4,
                    enableLearning=0,
                    enableDelay=False,
                    numDelayBits=0,
                    numTagBits=8, numWeightBits=8,
                    # weightLimitMant=10, weigthLimitExp=4,
                    signMode=nx.SYNAPSE_SIGN_MODE.MIXED)
            if pIdx == 0:

                if self.isConv:
                    # forWgts = np.ones((self.numHidNurns[pIdx], self.numMCs), int)*4
                    # numConvNs = self.lastConvLayer.sizeX * self.lastConvLayer.sizeY * self.lastConvLayer.sizeC
                    numConvNs = self.convNumInputs
                    if len(wgt) == 0:
                        forWgts = init_wgts(self.negwgtrng, self.poswgtrng, self.numHidNurns[pIdx], numConvNs, pIdx)
                        # forWgts = np.random.randint(low=self.negwgtrng, high=self.poswgtrng, size=(self.numHidNurns[pIdx], self.numMCs), dtype=int)
                    else:
                        forWgts = wgt[pIdx]

                    forConnGrp = self.net.createConnectionGroup(
                        src=self.lastConvLayer,
                        dst=self.allGCsPerPattern[pIdx],
                        prototype=forwardConnProtoBox,
                        weight=forWgts)

                    self.forwardConns[pIdx] = forConnGrp
                else:
                    # forWgts = np.ones((self.numHidNurns[pIdx], self.numMCs), int)*4
                    if len(wgt) == 0:
                        forWgts = init_wgts(self.negwgtrng, self.poswgtrng, self.numHidNurns[pIdx], self.numMCs, pIdx)
                        # forWgts = np.random.randint(low=self.negwgtrng, high=self.poswgtrng, size=(self.numHidNurns[pIdx], self.numMCs), dtype=int)
                    else:
                        forWgts = wgt[pIdx]

                    forConnGrp = self.net.createConnectionGroup(
                        src=self.allMCSomaGrp.soma,
                        dst=self.allGCsPerPattern[pIdx],
                        prototype=forwardConnProtoBox,
                        weight=forWgts)

                    self.forwardConns[pIdx] = forConnGrp


            else:
                if len(wgt) == 0:
                    forWgts = init_wgts(self.negwgtrng, self.poswgtrng, self.numHidNurns[pIdx],
                                        self.numHidNurns[pIdx - 1], pIdx)
                else:
                    forWgts = wgt[pIdx]

                #                 print(forWgts)

                forConnGrp = self.net.createConnectionGroup(
                    src=self.allGCsPerPattern[pIdx - 1],
                    dst=self.allGCsPerPattern[pIdx],
                    prototype=forwardConnProtoBox,
                    weight=forWgts)

                self.forwardConns[pIdx] = forConnGrp

    def connectbackwardConns(self, bwgt):
        """ creates the GC->MC inhibitory connections for each pattern"""
        ConnGroup = namedtuple("ConnGroup", "positive negative")

        # connections for the error path
        for pIdx in range(self.numlayers):

            if pIdx == 0:
                # self.backwardConns[pIdx] = self.net.createConnectionGroup()
                self.posbackwardConns[pIdx] = []
                self.negbackwardConns[pIdx] = []
                self.posbackwardConns[pIdx + 1] = []
                self.negbackwardConns[pIdx + 1] = []

            elif pIdx == self.numlayers - 1:
                np.random.seed(pIdx + 10)
                backwardConnProtoBox = nx.ConnectionPrototype(
                    numWeightBits=8,
                    signMode=nx.SYNAPSE_SIGN_MODE.MIXED,
                )

                if len(bwgt) == 0:
                    # posbackWgts = init_wgts(self.negwgtrng, self.poswgtrng, self.numHidNurns[pIdx-1], self.numHidNurns[pIdx])
                    posbackWgts = np.random.randint(low=self.bnegwgtrng, high=self.bposwgtrng,
                                                    size=(self.numHidNurns[pIdx - 1], self.numHidNurns[pIdx]),
                                                    dtype=int)
                else:
                    posbackWgts = bwgt[pIdx].T

                negbackWgts = - posbackWgts

                posbackConnGrp = self.net.createConnectionGroup(
                    src=self.allPosECsPerPattern[pIdx],
                    dst=self.allPosECsPerPattern[pIdx - 1].soma,
                    prototype=backwardConnProtoBox,
                    weight=posbackWgts)

                self.posbackwardConns[2 * pIdx] = posbackConnGrp

                posbackConnGrp = self.net.createConnectionGroup(
                    src=self.allNegECsPerPattern[pIdx],
                    dst=self.allPosECsPerPattern[pIdx - 1].soma,
                    prototype=backwardConnProtoBox,
                    weight=negbackWgts)

                self.posbackwardConns[2 * pIdx + 1] = posbackConnGrp

                negbackConnGrp = self.net.createConnectionGroup(
                    src=self.allNegECsPerPattern[pIdx],
                    dst=self.allNegECsPerPattern[pIdx - 1].soma,
                    prototype=backwardConnProtoBox,
                    weight=posbackWgts)

                self.negbackwardConns[2 * pIdx] = negbackConnGrp

                negbackConnGrp = self.net.createConnectionGroup(
                    src=self.allPosECsPerPattern[pIdx],
                    dst=self.allNegECsPerPattern[pIdx - 1].soma,
                    prototype=backwardConnProtoBox,
                    weight=negbackWgts)

                self.negbackwardConns[2 * pIdx + 1] = negbackConnGrp

            else:
                np.random.seed(pIdx + 10)
                backwardConnProtoBox = nx.ConnectionPrototype(
                    numWeightBits=8,
                    signMode=nx.SYNAPSE_SIGN_MODE.MIXED,
                )

                if len(bwgt) == 0:
                    # posbackWgts = init_wgts(self.negwgtrng, self.poswgtrng, self.numHidNurns[pIdx - 1], self.numHidNurns[pIdx])
                    posbackWgts = np.random.randint(low=self.bnegwgtrng, high=self.bposwgtrng,
                                                    size=(self.numHidNurns[pIdx - 1], self.numHidNurns[pIdx]),
                                                    dtype=int)
                else:
                    posbackWgts = bwgt[pIdx].T

                negbackWgts = - posbackWgts

                posbackConnGrp = self.net.createConnectionGroup(
                    src=self.allPosECsPerPattern[pIdx].soma,
                    dst=self.allPosECsPerPattern[pIdx - 1].soma,
                    prototype=backwardConnProtoBox,
                    weight=posbackWgts)

                self.posbackwardConns[2 * pIdx] = posbackConnGrp

                posbackConnGrp = self.net.createConnectionGroup(
                    src=self.allNegECsPerPattern[pIdx].soma,
                    dst=self.allPosECsPerPattern[pIdx - 1].soma,
                    prototype=backwardConnProtoBox,
                    weight=negbackWgts)

                self.posbackwardConns[2 * pIdx + 1] = posbackConnGrp

                negbackConnGrp = self.net.createConnectionGroup(
                    src=self.allNegECsPerPattern[pIdx].soma,
                    dst=self.allNegECsPerPattern[pIdx - 1].soma,
                    prototype=backwardConnProtoBox,
                    weight=posbackWgts)

                self.negbackwardConns[2 * pIdx] = negbackConnGrp

                negbackConnGrp = self.net.createConnectionGroup(
                    src=self.allPosECsPerPattern[pIdx].soma,
                    dst=self.allNegECsPerPattern[pIdx - 1].soma,
                    prototype=backwardConnProtoBox,
                    weight=negbackWgts)

                self.negbackwardConns[2 * pIdx + 1] = negbackConnGrp

    @timer
    def compileAndGetBoard(self):
        """ compiles the network """
        self.board = nx.N2Compiler().compile(self.net)
        return self.board

    def setupCxProbes(self, train):
        """ sets up the MC and GC compartment probes"""

        a = 1

        if train == False:
            # self.setupMCProbes()
            # self.setupLabelProbes()
            self.setupGCProbes()

        # self.setupMCProbes() # uncomment these when you want to verify
        # self.setupLabelProbes()
        # self.setupGCProbes()
        # self.setupECProbes()

        self.setupForwardSynapseProbes()
        # self.setupBackwardSynapseProbes()

    def setupMCProbes(self):
        """ sets up MC soma spike probes """
        if self.useLMTSpikeCounters:
            pc = nx.SpikeProbeCondition(tStart=1000000)
        else:
            pc = None
        probeParams = [nx.ProbeParameter.COMPARTMENT_VOLTAGE,
                       nx.ProbeParameter.SPIKE]
        probeParams = [nx.ProbeParameter.SPIKE]
        self.allMCSomaProbes = self.allMCSomaGrp.soma.probe(probeParams, pc)
        # if self.convNumInputs == 0:
        #     self.allMCSomaProbes = self.allMCSomaGrp.soma.probe(probeParams, pc)
        # else:
        #     self.allMCSomaProbes = self.lastConvLayer.probe(probeParams, pc)

    def setupLabelProbes(self):
        """ sets up MC soma spike probes """
        if self.useLMTSpikeCounters:
            pc = nx.SpikeProbeCondition(tStart=1000000)
        else:
            pc = None

        probeParams = [nx.ProbeParameter.COMPARTMENT_VOLTAGE,
                       nx.ProbeParameter.SPIKE]
        probeParams = [nx.ProbeParameter.SPIKE]
        self.allLabelProbes = self.allLabelGrp.soma.probe(probeParams, pc)

    def setupGCProbes(self):
        """ setup GC compartment probes """
        probeParams = [nx.ProbeParameter.SPIKE, nx.ProbeParameter.COMPARTMENT_VOLTAGE]
        probeParams = [nx.ProbeParameter.SPIKE]  # comment this when you want to verify
        self.gcProbes = dict()
        for pIdx in range(self.numlayers):
            cx = self.allGCsPerPattern[pIdx]
            prb = cx.probe(probeParams)
            self.gcProbes[pIdx] = prb

    def setupECProbes(self):
        """ setup GC compartment probes """
        probeParams = [nx.ProbeParameter.SPIKE]
        self.ecProbes = []
        for pIdx in range(self.numlayers):
            if pIdx == self.numlayers - 1:
                cx = self.allPosECsPerPattern[pIdx]
                prb = cx.probe(probeParams)
                self.ecProbes.append(prb)
                cx = self.allNegECsPerPattern[pIdx]
                prb = cx.probe(probeParams)
                self.ecProbes.append(prb)
            else:
                cx = self.allPosECsPerPattern[pIdx].soma
                prb = cx.probe(probeParams)
                self.ecProbes.append(prb)
                cx = self.allPosECsPerPattern[pIdx].dendrites[0]
                prb = cx.probe(probeParams)
                self.ecProbes.append(prb)
                cx = self.allNegECsPerPattern[pIdx].soma
                prb = cx.probe(probeParams)
                self.ecProbes.append(prb)
                cx = self.allNegECsPerPattern[pIdx].dendrites[0]
                prb = cx.probe(probeParams)
                self.ecProbes.append(prb)

    def setupForwardSynapseProbes(self):
        """ setup the GC->MC synapse probes """
        self.forwardConnsProbes = list()
        prbCond = nx.IntervalProbeCondition(tStart=0, dt=2000 * self.phaseLength)
        # prbCond = nx.IntervalProbeCondition(tStart=0, dt=self.phaseLength) # uncomment this when you want to verify
        prbParams = [nx.ProbeParameter.SYNAPSE_WEIGHT,
                     nx.ProbeParameter.SYNAPSE_DELAY,
                     nx.ProbeParameter.SYNAPSE_TAG,
                     nx.ProbeParameter.PRE_TRACE_X1,
                     #                      nx.ProbeParameter.POST_TRACE_Y1
                     ]

        # nx.ProbeParameter.PRE_TRACE_X1]
        # # nx.ProbeParameter.POST_TRACE_Y1],
        prbParams = [nx.ProbeParameter.SYNAPSE_WEIGHT]  # comment this when you want to verify
        for idx in range(self.numlayers):
            ConnGrp = self.forwardConns[idx]
            posConnGrpPrb = ConnGrp.probe(prbParams,
                                          probeConditions=[prbCond] * len(prbParams))
            self.forwardConnsProbes.append(posConnGrpPrb)

    def setupBackwardSynapseProbes(self):
        """ setup the GC->MC synapse probes """
        self.backwardConnsProbes = list()
        prbCond = nx.IntervalProbeCondition(tStart=0, dt=10 * 2 ** 5)
        # prbParams = [nx.ProbeParameter.SYNAPSE_WEIGHT,
        #              nx.ProbeParameter.SYNAPSE_DELAY,
        #              nx.ProbeParameter.SYNAPSE_TAG]
        # nx.ProbeParameter.POST_TRACE_Y1],
        prbParams = [nx.ProbeParameter.SYNAPSE_WEIGHT]
        for idx in range(self.numlayers):
            if idx > 0:
                ConnGrp = self.posbackwardConns[2 * idx]
                posConnGrpPrb = ConnGrp.probe([prbParams[0]],
                                              probeConditions=[prbCond] * len([prbParams[0]]))
                self.backwardConnsProbes.append(posConnGrpPrb[0])

                ConnGrp = self.posbackwardConns[2 * idx + 1]
                posConnGrpPrb = ConnGrp.probe([prbParams[0]],
                                              probeConditions=[prbCond] * len([prbParams[0]]))
                self.backwardConnsProbes.append(posConnGrpPrb[0])

                ConnGrp = self.negbackwardConns[2 * idx + 1]
                negConnGrpPrb = ConnGrp.probe([prbParams[0]],
                                              probeConditions=[prbCond] * len([prbParams[0]]))
                self.backwardConnsProbes.append(negConnGrpPrb[0])

                ConnGrp = self.negbackwardConns[2 * idx + 1]
                negConnGrpPrb = ConnGrp.probe([prbParams[0]],
                                              probeConditions=[prbCond] * len([prbParams[0]]))
                self.backwardConnsProbes.append(negConnGrpPrb[0])

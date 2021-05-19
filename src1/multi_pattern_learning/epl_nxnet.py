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

        ### create convolutional layers #must match the conv layers in the saved keras

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
            print("ethreshold = ")
            print(ethold)
        else:
            wsrc = self.numHidNurns[patternIdx - 1]
            thold = init_th(wsrc, patternIdx, self.hid_vth, 255)
            self.biasMn = int(thold / 10)
            ref = 1
            ethold = int(self.LabeltoECwgt * (self.numHidNurns[patternIdx + 1] / 2.0))
            print("ethreshold = ")
            print(ethold)


        numTmps = int(np.ceil(thold / 255))
        numTmpsNeurons = 1

        print("threshold = ")
        print(thold)


        print("bias Mantissa: " , self.biasMn)
        print("bias Exponent: ", self.biasEx)

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

        if patternIdx == self.numlayers - 1:
            for i in range(self.numHidNurns[patternIdx]):
                tauU = 2
                tauV = 25
                decayU = int(1 / tauU * 2 ** 12)
                decayV = int(1 / tauV * 2 ** 12
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
                    vThMant=2,  # i.e. 2 * 64 = 128
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
                GCtoWtaConnProtoBox = nx.ConnectionPrototype(
                    weight=self.LabeltoECwgt, # i.e. 8
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
                numWeightBits=8,            # if patternIdx == self.numlayers - 1:
            )

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


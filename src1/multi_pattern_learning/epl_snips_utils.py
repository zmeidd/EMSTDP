# - adapted from EPL module - NxSDKapp 0.9.5rc
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import os
import inspect
from .epl_nxnet import EplNxNet, timer
from jinja2 import Environment, FileSystemLoader


class EplWithSNIPs(EplNxNet):
    """ This class has all the functions/methods to setup, gather and transfer
    the data required by the LMT SNIPs code which manages the execution of the
    EPL network"""

    def __init__(self, eplParams):
        super().__init__(eplParams=eplParams)
        self.totalTimeSteps = None
        self.totalTestSamples = None
        self.lenData = 1 + 2 + 4 * self.numInputs

    def _updateLength(self, axonMap):
        """ updates the length of the data being sent """
        for l in axonMap.values():
            self.lenData += len(l)

    @staticmethod
    def _printMap(mapp):
        """ display the inputAxonIds for each core"""
        for ky in sorted(mapp.keys()):
            print("{}=>{}".format(ky, sorted(mapp[ky])))
            print(len(mapp[ky]))

    def _updateMap(self, connGrp, coreIdToAxonIdMap):
        """ updates the input axon ids for each core for a given connection
               group """
        for conn in connGrp:
            iaxonId = conn.inputAxon.nodeId
            hwAxonIds = self.net.resourceMap.inputAxon(iaxonId)
            for _, _, coreId, axonId in hwAxonIds:
                if coreId not in coreIdToAxonIdMap.keys():
                    coreIdToAxonIdMap[coreId] = set()
                coreIdToAxonIdMap[coreId].add(axonId)

    def getCoreIdToAxonIdMapForForwardConns(self):
        """ creates a mapping of input axon ids for each core for the MC->GC
                connections """
        coreIdToAxonIdMap = dict()
        for connGrp in self.forwardConns.values():
            self._updateMap(connGrp, coreIdToAxonIdMap)
        # pp.pprint(coreIdToAxonIdMap)
        self.forwardInputAxonMap = coreIdToAxonIdMap
        self._printMap(self.forwardInputAxonMap)
        self._updateLength(self.forwardInputAxonMap)

    def getCoreIdToAxonIdMapForBackwardConns(self):
        """ creates a mapping of input axon ids for each core for the GC->MC
                connections """
        self.backwardConnInputAxonMap = dict()
        # self.gcToMCInhConnInputAxonMap = dict()
        for connGrp in self.backwardConns.values():
            self.updateMap(connGrp, self.backwardConnInputAxonMap)
            #self.updateMap(connGrp.negative, self.gcToMCInhConnInputAxonMap)
        self._updateLength(self.backwardConnInputAxonMap)
        self.printMap(self.backwardConnInputAxonMap)

    def forwardAxonIdMapPerPattern(self, patternIdx):
        """ creates a mapping of input axon ids for each core for the GC->MC
            connections belonging to only a particular pattern """
        inputAxonIdMap = dict()
        # for colIdx in range(self.numInputs):
        connGrp = self.forwardConns[patternIdx]
        self.updateMap(connGrp, inputAxonIdMap)
        return inputAxonIdMap

    def backwardAxonIdMapPerPattern(self, patternIdx):
        """ creates a mapping of input axon ids for each core for the GC->MC
            connections belonging to only a particular pattern """
        inputAxonIdMap = dict()
        # for colIdx in range(self.numInputs):
        connGrp = self.backwardConns[patternIdx]
        self.updateMap(connGrp, inputAxonIdMap)
        return inputAxonIdMap

    @timer
    def _genCodeForConstants(self):
        """ write the constants (known before SNIP compilation) to a
                C header file using Jinja2"""
        cfile = self.snipsDir + "/constants.h"
        templatesDir = self.currDir + "/templates"
        env = Environment(loader=FileSystemLoader(templatesDir))
        template = env.get_template('constants_template.txt')
        consts = dict()
        consts['TRAIN'] = self.trainbool
        consts['NUM_CORES'] = self.lastUsedLogicalCoreId + 1
        consts['NUM_PATTERNS'] = self.numlayers
        consts['NUM_MCS'] = self.numInputs
        consts['NUM_GCS'] = sum(self.numHidNurns)

        consts['NUM_GC_CORES'] = len(sorted(list(set(self._physicalCoreId))))
        consts['NUM_TARGETS'] = self.numTargets
        consts['MCSOMA_CXGRP_ID'] = self.allMCSomaGrp.dendrites[0].groupId
        consts['LABEL_CXGRP_ID'] = self.allLabelGrp.dendrites[0].groupId
        consts['LAST_POSECLAYER_CXGRP_ID'] = self.allPosECsPerPattern[self.numlayers-1].groupId
        consts['LAST_NEGECLAYER_CXGRP_ID'] = self.allNegECsPerPattern[self.numlayers - 1].groupId
        consts['GAMMA_CYCLE_DURATION'] = self.phaseLength

        consts['NO_LEARNING_PERIOD'] = 20
        consts['NUM_TRAIN_SAMPLES'] = self.totalTrainSamples
        consts['NUM_TEST_SAMPLES'] = self.totalTestSamples
        consts['USE_LMT_SPIKE_COUNTERS'] = 1 if self.useLMTSpikeCounters else 0
        consts['RUN_TIME'] = self.totalTimeSteps
        consts['LOG_SNIP'] = 1 if self.logSNIPs else 0
        consts['CONV_GC_CORE_ID_BEGIN'] = self.gcCoreIdRange[0]
        consts['GC_CORE_ID_BEGIN'] = self.gcCoreIdRange[1]
        consts['GC_CORE_ID_END'] = self.gcCoreIdRange[2]
        consts['LABEL_CORE_ID'] = self.gcCoreIdRange[2]+1
        gcGrpsPerPatternId = [grp.groupId for grp in
                              self.allGCsPerPattern.values()]
        # output = template.render(data=consts, grpIds=gcGrpsPerPatternId)

        gcGrpsPerPatternId1 = [grp for grp in
                              self.numHidNurns]

        gcGrpsPerPatternId2 = sorted(list(set(self._physicalCoreId)))

        gcGrpsPerPatternId3 = self._physicalCoreId

        gcGrpsPerPatternId4 = self._stdpPostStateIndex

        gcGrpsPerPatternId5 = self._stdpProfile

        gcGrpsPerPatternId6 = self._traceProfile

        # gcGrpsPerPatternId7 = self.dtrite

        output = template.render(data=consts, grpIds=gcGrpsPerPatternId, grpIds1=gcGrpsPerPatternId1,
                                 grpIds2=gcGrpsPerPatternId2, grpIds3=gcGrpsPerPatternId3,
                                 grpIds4=gcGrpsPerPatternId4, grpIds5=gcGrpsPerPatternId5, grpIds6=gcGrpsPerPatternId6)
                                 # , grpIds7=gcGrpsPerPatternId7)
        with open(cfile, "w") as fh:
            fh.write(output)

    def _determineCoreIdCompartmentIdTraceState(self, compartmentSet, net, board):
        """Determine the board registers which must be overwritten, as well as their initial state

        :param compartmentGroup compartmentGroup: The group of compartments to overwrite
        :param NxNet net: the net containing the compartmentGroup
        :param N2Board board: the compiled board
        :return list(int) physicalCoreId: the core numbers to modify
        :return list(int) stdPostStateIndex: the STDP post state registers to modify
        :return list(int) stdpProfile: initial values of the stdpProfile field of the STDP post state registers
        :return list(int) traceProfile: initial values of the traceProfile field of the STDP post state registers
        """

        # allocate some memory to hold the coreId and compartmentIndex
        # these values will be communicated to the snip at initialization



        physicalCoreId = [int]*sum(self.numHidNurns)
        stdpPostStateIndex = [int]*sum(self.numHidNurns)
        stdpProfile = [int]*sum(self.numHidNurns)
        traceProfile = [int]*sum(self.numHidNurns)

        # loop through the compartments and find the physical core and compartment for each
        cpts = 0
        for sa in range(len(compartmentSet)):
            compartmentGroup = compartmentSet[sa]

            for cc in range(compartmentGroup.numNodes):
                boarId, chipId, coreId, compartmentId, _, _ = net.resourceMap.compartment(
                    compartmentGroup[cc].nodeId)
                # physicalCoreId[cc +cpts] = board.n2Chips[chipId].n2Cores[coreId].id

                physicalCoreId[cc + cpts] = coreId

                # the stdpPostStateIndex is the compartment entryId
                stdpPostStateIndex[cc +cpts] = int(compartmentId)
                stdpProfile[cc +cpts] = board.n2Chips[chipId].n2Cores[coreId].stdpPostState[compartmentId].stdpProfile
                traceProfile[cc +cpts] = board.n2Chips[chipId].n2Cores[coreId].stdpPostState[compartmentId].traceProfile

            cpts += compartmentGroup.numNodes

        return physicalCoreId, stdpPostStateIndex, stdpProfile, traceProfile

    def _initSnip(self):
        """ setups the init SNIP"""
        includeDir = self.snipsDir
        cFilePath = includeDir + "/initsnip.c"
        self.initProcess = self.board.createProcess("self.initProcess",
                                                    includeDir=includeDir,
                                                    cFilePath=cFilePath,
                                                    funcName="initParamsAndInputs",
                                                    guardName=None,
                                                    phase="init")
        # lenData = len(self.initData) + self.numMCs + 1
        lenData = self.numMCs + self.numTargets + 0
        # lenData = 1
        self.initChannel = self.board.createChannel(b'nxinit', messageSize=4, numElements=lenData)
        self.initChannel.connect(None, self.initProcess)

        # numTrainCompartments = sum(self.numHidNurns)
        # self._numTrainCompartments = numTrainCompartments
        #
        # # communicates which coreIds the compartments lie on
        # # what is the limit on the channel size here?
        # self._initCoreIdsChannel = self.board.createChannel(
        #     b'nxinitCoreIds', messageSize=4, numElements=numTrainCompartments)
        # self._initCoreIdsChannel.connect(None, self.initProcess)
        #
        # # communicates which compartments (within the core) to modify
        # # what is the limit on the channel size here?
        # self._initStdpCompartmentIndexChannel = self.board.createChannel(
        #     b'nxinitStdpCompartmentIndex', messageSize=4, numElements=numTrainCompartments)
        # self._initStdpCompartmentIndexChannel.connect(None, self.initProcess)
        #
        # # communicates the initial TraceProfile values
        # self._initStdpProfileChannel = self.board.createChannel(
        #     b'nxinitStdpProfile', messageSize=4, numElements=numTrainCompartments)
        # self._initStdpProfileChannel.connect(None, self.initProcess)
        #
        # # communicates which compartments (within the core) to modify
        # # what is the limit on the channel size here?
        # self._initTraceProfileChannel = self.board.createChannel(
        #     b'nxinitTraceProfile', messageSize=4, numElements=numTrainCompartments)
        # self._initTraceProfileChannel.connect(None, self.initProcess)

    def _mgmtSnip(self):
        """ setups up the management SNIP """
        includeDir = self.snipsDir
        cFilePath = includeDir + "/mgmtsnip.c"
        self.mgmtProcess = self.board.createProcess("self.mgmtProcess",
                                                    includeDir=includeDir,
                                                    cFilePath=cFilePath,
                                                    funcName="runMgmt",
                                                    guardName="doMgmt",
                                                    phase="mgmt")
        # lendata = self.lenData * 2
        # self.inputAxonIdDataMgmtChannel = self.board.createChannel(
        #     b'nxmgmt_input_axon_ids', "int", lendata)
        # self.inputAxonIdDataMgmtChannel.connect(None, self.mgmtProcess)
        lenMCInputs = (self.totalTrainSamples + self.totalTestSamples)
        lenMCInputs *= (self.numMCs)
        self.mcInputsMgmtChannel = self.board.createChannel(
            b'nxmgmt_mc_inputs', messageSize=4, numElements=lenMCInputs)
        self.mcInputsMgmtChannel.connect(None, self.mgmtProcess)

        # lenMCInputs = (self.totalTrainSamples + self.totalTestSamples - 1)
        # lenMCInputs *= (10)
        # self.mcLabelsMgmtChannel = self.board.createChannel(
        #     b'nxmgmt_mc_labels', "int", lenMCInputs)
        # self.mcLabelsMgmtChannel.connect(None, self.mgmtProcess)

        lenMCInputs = (self.totalTrainSamples + self.totalTestSamples)
        lenMCInputs *= self.numTargets*2
        self.labelInputsMgmtChannel = self.board.createChannel(
            b'nxmgmt_label_inputs', messageSize=4, numElements=lenMCInputs)
        self.labelInputsMgmtChannel.connect(None, self.mgmtProcess)

        # self.spkCounterMgmtChannel = self.board.createChannel(
        #     b'nxspkcntr', "int", 2 ** 18)
        # self.spkCounterMgmtChannel.connect(self.mgmtProcess, None)

        # self.boardStatusMgmtChannel = self.board.createChannel(
        #     b'status', "int", 10)
        # self.boardStatusMgmtChannel.connect(self.mgmtProcess, None)

    def idxToBases(self, inputList):
        """ maps the input data/sensor reading to an MC-AD bias current"""
        return [self.stim2bias[i] for i in inputList]

    # def _sendInputAxonMapData(self, axonMap):
    #     """ send the input axon map data to mgmt SNIP via a read channel"""
    #     keys = sorted(axonMap.keys())
    #     self.inputAxonIdDataMgmtChannel.write(1, [len(keys)])
    #     for coreId in keys:
    #         data = list(axonMap[coreId])
    #         # print(data)
    #         self.inputAxonIdDataMgmtChannel.write(2, [coreId, len(data)])
    #         self.inputAxonIdDataMgmtChannel.write(len(data), data)
    #
    # def _sendDataToSwitchToInference(self):
    #     """ send the data required to switch from training mode to inference
    #         mode by the mgmt SNIP via a read channel"""
    #     mode = 1  # testing
    #     self.inputAxonIdDataMgmtChannel.write(1, [mode])
    #     self._sendInputAxonMapData(axonMap=self.forwardInputAxonMap)
    #     self._sendInputAxonMapData(axonMap=self.backwardConnInputAxonMap)

    def sendInitDataToSNIP(self):
        """ send the data needed for the init and mgmt SNIPs"""

        # lt = len(self.trainData)
        # lb = len(self.trainLabel)
        tmp = []
        tmp += list([128 for j in range(self.numMCs)])
        tmp += list([64 for j in range(self.numTargets)])
        self.initChannel.write(self.numMCs + self.numTargets, tmp)

        # self._initCoreIdsChannel.write(
        #     self._numTrainCompartments, self._physicalCoreId)
        #
        # self._initStdpCompartmentIndexChannel.write(
        #     self._numTrainCompartments, self._stdpPostStateIndex)
        #
        # self._initStdpProfileChannel.write(
        #     self._numTrainCompartments, self._stdpProfile)
        #
        # self._initTraceProfileChannel.write(
        #     self._numTrainCompartments, self._traceProfile)

    def sendDataToSNIP(self, start, end, trainperiod=True):
        """ send the data needed for the init and mgmt SNIPs"""

        if trainperiod:
            for i in range(start, end, 1):
                self.labelInputsMgmtChannel.write((self.numTargets*2),
                                               self.trainLabel[(i) * (self.numTargets*2):(i) * (self.numTargets*2) + self.numTargets*2])
                self.mcInputsMgmtChannel.write((self.numMCs),
                                               self.trainData[(i)*(self.numMCs):(i)*(self.numMCs)+self.numMCs])

        # self.mcInputsMgmtChannel.write(len(self.trainData), self.trainData)
        # self.labelInputsMgmtChannel.write(len(self.trainLabel), self.trainLabel)
        else:
            for i in range(start, end, 1):
                self.labelInputsMgmtChannel.write((self.numTargets*2),
                                               self.testLabel[(i) * (self.numTargets*2):(i) * (self.numTargets*2) + self.numTargets*2])
                self.mcInputsMgmtChannel.write((self.numMCs),
                                               self.testData[(i)*(self.numMCs):(i)*(self.numMCs)+self.numMCs])

        # for i in range(self.totalTestSamples):
        #     self.mcInputsMgmtChannel.write((self.numMCs),
        #                                    self.testData[(i)*(self.numMCs):(i)*(self.numMCs)+self.numMCs])
        #
        # # self.mcInputsMgmtChannel.write(len(self.testData), self.testData)
        # self.labelInputsMgmtChannel.write(len(self.testLabel), self.testLabel)

    def gatherDataForSNIPs(self, trainingSet, testingSet, train):
        """ collect all the data that needs to be sent to the SNIPs via
                channels"""

        self._physicalCoreId, self._stdpPostStateIndex, self._stdpProfile, self._traceProfile = \
            self._determineCoreIdCompartmentIdTraceState(self.allGCsPerPattern, self.net, self.board)

        # self.getCoreIdToAxonIdMapForForwardConns()

        if train:

            self.trainData = []
            self.trainLabel = []
            # self.trainData += self.idxToBases(trainingSet[0][0])
            # self.trainLabel += trainingSet[1][0]
            # self.trainLabel += trainingSet[1][0]
            for trainSample in range(len(trainingSet[0])):
                self.trainData += self.idxToBases(trainingSet[0][trainSample])
                # self.trainLabel += trainingSet[1][trainSample]
                self.trainLabel += list([0 for j in range(self.numTargets)])
                self.trainLabel += trainingSet[1][trainSample]

        else:
            self.testData = []
            self.testLabel = []
            for testSample in range(len(testingSet[0])):
                self.testData += self.idxToBases(testingSet[0][testSample])
                self.testLabel += list([0 for j in range(self.numTargets)])
                
    def addSNIPs(self, totalTrainSamples, totalTestSamples, totalTimeSteps):
        """ create and configure the SNIPs """
        self.totalTrainSamples = totalTrainSamples
        self.totalTestSamples = totalTestSamples
        self.totalTimeSteps = totalTimeSteps
        self.currDir = os.path.dirname(os.path.abspath(inspect.getfile(
            inspect.currentframe())))
        self.snipsDir = self.currDir + "/snips"
        self._genCodeForConstants()
        self._initSnip()
        self._mgmtSnip()

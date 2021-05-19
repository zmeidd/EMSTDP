# - adapted from EPL module - NxSDKapp 0.9.5rc
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import os
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import nxsdk.api.n2a as nx
from .epl_snips_utils import EplWithSNIPs
from .epl_parameters import ParamemtersForEPL
from .epl_utils import EplUtils
from matplotlib.lines import Line2D


class EPLMultiPatternLearning(EplUtils, EplWithSNIPs):
    """
    Create an EPL network to learn multiple patterns and test if the network
    can recall the learned patterns even when noise corrupted test samples of
    the learned patterns are presented
    """

    def __init__(self, eplParams, train, conv_wgt, wgt, bwgt):
        """ initialize the EPL network """
        super().__init__(eplParams=eplParams)
        self.setupNetwork(train, conv_wgt, wgt, bwgt)
        self.setupCxProbes(train)
        self.board = self.compileAndGetBoard()
        self.numStepsRan = None
        self.trainingSet = None
        self.testingSet = None
        self.cxIdToSpikeTimeMap = None
        self.train = train
        self.wgt = wgt

    @property
    def numTrainSamples(self):
        """ returns the number of training patterns presented to the network"""
        return len(self.trainingSet[0])

    @property
    def numTestSamples(self):
        """ returns the number of noise corrupted test samples for
        each pattern presented to the network"""
        return len(self.testingSet[0])

    @property
    def numLabelSamples(self):
        """ returns the number of patterns presented in the labeling phase.
        In the labeling phase, the learned patters are presented as test
        samples"""
        return len(self.trainingSet[0])

    @property
    def numTotalTestSamples(self):
        """ returns the total number of the samples presented to the network
        in test phase. The first sample is always the learned odor (labeling
        phase). Thus labeling is testing without any noise corruption i.e.
        the trained odor. Next in the actual test phase, we present various
        noise corrupted versions of the trained odor"""
        # return self.numTestSamples + self.numLabelSamples
        return self.numTestSamples

    @property
    def trainDuration(self):
        """ returns the total number of timesteps in the train phase """
        gamma = self.phaseLength
        # positiveThetaPeriod = self.numGammaCyclesTrain * gamma
        # negativeThetaPeriod = self.numGammaCyclesIdle * gamma
        trainingDuration = 2*gamma
        trainingDuration *= self.numTrainSamples
        return trainingDuration-0

    @property
    def testDuration(self):
        """ returns the total number of timesteps in the test phase """
        gamma = self.phaseLength
        # negativeThetaPeriod = self.numGammaCyclesIdle * gamma
        testDuration = 2 * gamma
        # testDuration += negativeThetaPeriod
        testDuration *= self.numTotalTestSamples
        return testDuration

    @property
    def numStepsToRun(self):
        """ returns the total number of steps for which the network will run"""
        return self.trainDuration + self.testDuration

    def _readSpikeCounterData(self):
        """ post process the LMT spike counter data sent from the chip """
        # ToDo: remove magic numbers, use logger
        print("processing LMT spike counter data...takes time...")
        cxIdToTimeMap = dict()
        for cx in range(self.numMCs):
            cxIdToTimeMap[cx] = list()

        while True:
            t = self.spkCounterMgmtChannel.read(1)[0]
            #print("Processing spike at t={}".format(t))
            cx = -1
            while True:
                cx = self.spkCounterMgmtChannel.read(1)[0]
                #print("Processing for cx={}".format(cx))
                if cx >= self.numStepsToRun + 10:
                    break
                else:
                    cxIdToTimeMap[cx].append(t)
            if cx == self.numStepsToRun + 11:
                break
        # pprint.pprint(cxIdToTimeMap)
        return cxIdToTimeMap

    def genTrainingData(self, numlayers=1, datset="mnist"):
        """generates a synthetic training dataset of odor sensor readings"""
        # return self.generateTrainingData(numOdors=numlayers,
        #                                  numSensors=self.numMCs)
        return self.getTrainingData(numSamples=numlayers, datset=datset)

    def genTestingData(self, numlayers=1, datset="mnist"):
        """generates a synthetic testing dataset of noise corrupted
        versions of odor sensor readings"""
        # return self.generateTestingData(trainingData=trainingData,
        #                                 occlusionPercent=occlusionFactor,
        #                                 numTestSamples=numTestSamples)
        return self.getTestingData(numSamples=numlayers, datset=datset)

    def fit(self, trainingSet, epochs=1, atatime=1000):
        """trains the network with the given training set"""
        self.trainingSet = trainingSet
        self.testingSet = [[]]
        self.gatherDataForSNIPs(trainingSet,self.testingSet, self.train)
        self.addSNIPs(totalTrainSamples=self.numTrainSamples,
                      totalTestSamples=0,
                      totalTimeSteps=self.numStepsToRun)

        # self.board.startDriver()
        self.board.start()
        self.sendInitDataToSNIP()
        singlerun = 2*self.phaseLength
        self.board.run(singlerun, aSync=False)

        # self.sendDataToSNIP(0, self.numTrainSamples, trainperiod=True)
        # self.board.run(self.numStepsToRun+singlerun, aSync=True)

        # trainperiod = True
        for j in range(epochs):
            gp = atatime
            singleTP = gp*2*self.phaseLength
            for i in range(0, self.numTrainSamples, gp):
                print("Epoch: ", j)
                print("training sample: ", i, " -> ", i+gp)
                self.sendDataToSNIP(i, i+gp, trainperiod=self.train)
                self.board.run(singleTP, aSync=False)

        # SNIP halts after training; must do this read to proceed to testing
        # status = self.boardStatusMgmtChannel.read(1)[0]
        self.board.finishRun()
        self.board.disconnect()

    def test(self, testingSet, epochs=1, atatime=1000):
        """trains the network with the given training set"""
        self.trainingSet = [[]]
        self.testingSet = testingSet
        self.gatherDataForSNIPs(self.trainingSet, testingSet, self.train)
        self.addSNIPs(totalTrainSamples=0,
                      totalTestSamples=self.numTotalTestSamples,
                      totalTimeSteps=self.numStepsToRun)

        # self.board.startDriver()
        self.board.start()
        self.sendInitDataToSNIP()
        singlerun = 2 * self.phaseLength
        self.board.run(singlerun, aSync=False)

        # self.sendDataToSNIP(0, self.numTrainSamples, trainperiod=True)
        # self.board.run(self.numStepsToRun+singlerun, aSync=True)

        # trainperiod = True
        for j in range(epochs):
            gp = atatime
            singleTP = gp * 2 * self.phaseLength
            for i in range(0, self.numTotalTestSamples, gp):
                print("testing sample: ", i, " -> ", i+gp)
                self.sendDataToSNIP(i, i + gp, trainperiod=self.train)
                self.board.run(singleTP, aSync=False)

        # SNIP halts after training; must do this read to proceed to testing
        # status = self.boardStatusMgmtChannel.read(1)[0]
        self.board.finishRun()
        self.board.disconnect()


    def predict(self):
        """tests the network with the given testing set"""
        self.board.run(self.testDuration, aSync=True)
        if self.useLMTSpikeCounters:
            self.cxIdToSpikeTimeMap = self._readSpikeCounterData()
        self.board.finishRun()
        self.board.disconnect()

    def evaluate(self, verbose=False, similarityThreshold=0.75):
        """evaluates the performance of the network"""

        spikesData = self.cxIdToSpikeTimeMap if self.useLMTSpikeCounters else \
            self.allMCSomaProbes
        gammaCode = self.spikesDataToGammaCode(
            spikesData=spikesData,
            numStepsRan=self.numStepsToRun,
            numSensors=self.numMCs,
            cycleDuration=self.phaseLength,
            useLMTCounters=self.useLMTSpikeCounters,
            dumpToFile=False)

        nGammaTrain = self.numGammaCyclesTrain + self.numGammaCyclesIdle
        accuracy_percent = self.computeResults(
            gammaCode=gammaCode,
            nGammaPerTraining=nGammaTrain,
            trainingSetSize=len(self.trainingSet),
            testSetSize=len(self.testingSet),
            verbose=verbose, nsensors=self.numMCs,
            similarityThreshold=similarityThreshold)
        return accuracy_percent

    def showRasterPlot(self, sampleIdx):
        """ displays the MC output spike raster during the test phase for a
        particular test sample (sampleIdx) belonging to a particular pattern(
        patternIdx)"""
        inprobes = self.allMCSomaProbes
        labprobes = self.allLabelProbes
        testDuration = self.phaseLength * 2

        beginLabel = self.trainDuration + (sampleIdx * testDuration)
        endLabel = beginLabel + testDuration
        dataLabel = [np.nonzero(probe.data[beginLabel:endLabel])[0]
                            for probe in labprobes]

        # beginSample = self.trainDuration + (self.numlayers * testDuration)
        # beginSample += (patternIdx * self.numTestSamples + sampleIdx) * \
        #                testDuration
        # endSample = beginSample + testDuration
        dataSample = [np.nonzero(probe.data[beginLabel:endLabel])[0]
                     for probe in inprobes]

        size = self.numMCs
        fig, _ = plt.subplots()
        plt.eventplot(positions=dataLabel, colors='blue',
                      lineoffsets=np.arange(size),
                      linelengths=0.8)
        plt.eventplot(positions=dataSample, colors='red',
                      lineoffsets=np.arange(size),
                      linelengths=0.5)
        plt.title("""MC Output Spike Raster (sampleIdx={})
                    """.format(sampleIdx))
        plt.ylabel("#MC Neurons")
        plt.xlabel("""Time ({} gamma cycles; gamma cycle={} timesteps)
                """.format(self.numGammaCyclesTest, self.phaseLength))
        xticks = [self.phaseLength * i for i in
                                range(self.numGammaCyclesTest)]
        plt.xticks(xticks)
        legend_elements = [
            Line2D([0], [0], color='blue', lw=1, label='Trained Pattern'),
            Line2D([1], [0], color='red', lw=1, label='Test Pattern')
            ]
        fig.legend(handles=legend_elements, loc='center right')
        fig.tight_layout()
        fig.subplots_adjust(right=0.9)
        plt.show()


def test1(numTestSamples, useLMTSpkCtr=False):
    #os.environ["PARTITION"] = "wm_perf"
    # user specifies the dimensions, compute the cores internally
    eplParams = ParamemtersForEPL()
    eplParams.numlayers = 5
    eplParams.numInputs = 72
    eplParams.numMCsPerColumn = 1
    eplParams.numHidNurnsPerColumn = 5
    eplParams.connProbMCToGC = 0.2
    eplParams.numDelaysMCToGC = 1
    eplParams.useRandomSeed = True
    eplParams.randomGenSeed = 100
    eplParams.useLMTSpikeCounters = useLMTSpkCtr
    eplParams.logSNIPs = False
    eplParams.executionTimeProbe = False
    eplParams.numGammaCyclesTrain = 45

    epl = EPLMultiPatternLearning(eplParams=eplParams)

    x = epl.genTrainingData(numlayers=eplParams.numlayers)
    y = epl.genTestingData(trainingData=x,
                           numTestSamples=numTestSamples,
                           occlusionFactor=0.5)
    epl.fit(trainingSet=x, testingSet=y)
    epl.predict()
    epl.evaluate(verbose=True)
    epl.showRasterPlot(patternIdx=0, sampleIdx=2)


if __name__ == '__main__':
    test1(numTestSamples=5, useLMTSpkCtr=False)
    #test2()

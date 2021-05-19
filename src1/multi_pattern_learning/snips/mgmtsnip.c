/*
INTEL CONFIDENTIAL

Copyright © 2018 Intel Corporation.

This software and the related documents are Intel copyrighted
materials, and your use of them is governed by the express
license under which they were provided to you (License). Unless
the License provides otherwise, you may not use, modify, copy,
publish, distribute, disclose or transmit  this software or the
related documents without Intel's prior written permission.

This software and the related documents are provided as is, with
no express or implied warranties, other than those that are
expressly stated in the License.
*/

#include "mgmtsnip.h"

#include <time.h>
#include <stdlib.h>



#define FINISHED_TRAINING 1
//static int mgmtChannelId = INVALID_CHANNEL_ID;
static int mcInputsChannelId = INVALID_CHANNEL_ID;
static int labelInputsChannelId = INVALID_CHANNEL_ID;
//static int spikeCounterChannelId = INVALID_CHANNEL_ID;
//static int statusChannelId = INVALID_CHANNEL_ID;
//static int mcSomaSpikeCounters[NUM_MCS];
int resetmode = 0;
//static int excSynFmtId = -1;
//static int numTest = 0;
static int total_steps;
//static int numlayersLearned = 0;
//static int mcSomaNumSTDP = 0;



int doMgmt(runState *s) {

    if (USE_LMT_SPIKE_COUNTERS==1 && s->time_step==1){
//        spikeCounterChannelId = getChannelID("nxspkcntr");
        total_steps = RUN_TIME; //s->total_steps;
        //printf("*****TOTAL TIME STEPS = %d \n", total_steps);
    }

    if (s->time_step == 1) {
//        statusChannelId = getChannelID("status");
        mcInputsChannelId = getChannelID("nxmgmt_mc_inputs");
        labelInputsChannelId = getChannelID("nxmgmt_label_inputs");
    }

//    int stateDuration;
//    int timeElapsed = s->time_step - tbeginCurrState;
    int runCommand = 0;
    resetmode = 0;
//    LOG("MGMT %d: stateDuration %d, timeElapsed %d\n", s->time_step, stateDuration, timeElapsed);
    if (USE_LMT_SPIKE_COUNTERS) command = DO_NOTHING;

    if ((s->time_step+3) < (2*GAMMA_CYCLE_DURATION)){
        return runCommand;
    }

//    if ((s->time_step) % (2*GAMMA_CYCLE_DURATION*NUM_TRAIN_SAMPLES) == 0){
//        int tmp = changeMode(s);
//        if (tmp == 0)
//            {command = SWITCH_TO_POSITIVE_THETA_TESTING;}
//        if (tmp == 1)
//            {command = SWITCH_TO_POSITIVE_THETA_TRAINING;}
////        runCommand = 1;
//    }

    if (mode == TRAINING) {
        if ((s->time_step-2) % (2*GAMMA_CYCLE_DURATION) == 0){
            command = SWITCH_TO_POSITIVE_THETA_TRAINING;
            runCommand = 1;
        }
        else if ((s->time_step-1) % GAMMA_CYCLE_DURATION == 0){
            command = RESET;
            runCommand = 1;
            if ((s->time_step-1) % (2*GAMMA_CYCLE_DURATION) == 0){
                resetmode = 1;
            }
        }

    }

    if (mode == TESTING) {
        if ((s->time_step-2) % (2*GAMMA_CYCLE_DURATION) == 0){
            command = SWITCH_TO_POSITIVE_THETA_TESTING;
            runCommand = 1;
            resetmode = 1;
        }
        else if ((s->time_step-1) % (2*GAMMA_CYCLE_DURATION) == 0){
            command = RESET;
            runCommand = 1;
            resetmode = 1;
//            if ((s->time_step-1) % (2*GAMMA_CYCLE_DURATION) == 0){
//                resetmode = 1;
//            }
        }
    }


    if (USE_LMT_SPIKE_COUNTERS) return 1;
    else return runCommand;
}

//int readAndResetSpikeCounter(int probeId, int time_step) {
//    int idx = 0x20 + probeId;
//    int t = time_step - 1;
//    if (SPIKE_COUNT[t&3][idx] >= 1) {
//        SPIKE_COUNT[t&3][idx] = 0;
////        LOG("t=%d: cxID=%d spiked at  \n", t-1, probeId);
//        return 1;
//    }
//    else {
//        SPIKE_COUNT[t&3][idx] = 0;
//        return 0;
//    }
//}
//
//void updateSpikeCounters(int time_step) {
////    LOG("TIMESTEP = %d:updating spike counters...\n", time_step);
//    if (time_step == 1) return;
//    int t = time_step - 2;
//    int hasSpiked;
//    int marker = total_steps + 10;
//    writeChannel(spikeCounterChannelId, &t, 1);
//
//    for (int cxId=0; cxId < NUM_MCS; cxId++) {
//        hasSpiked = readAndResetSpikeCounter(cxId, time_step);
//        if (hasSpiked) writeChannel(spikeCounterChannelId, &cxId, 1);
//    }
//    if (time_step == total_steps) marker++;
//    writeChannel(spikeCounterChannelId, &marker, 1);
//}

//void initMCInputsChannel() {
//    if(mcInputsChannelId == INVALID_CHANNEL_ID) {
//        mcInputsChannelId = getChannelID("nxmgmt_mc_inputs");
//        if(mcInputsChannelId == INVALID_CHANNEL_ID) {
////            LOG("ERROR: Invalid channelID for nxmgmt_mc_inputs \n");
//        }
//    }
//
//}

//void initLabelChannel() {
//    if(labelInputsChannelId == INVALID_CHANNEL_ID) {
//        labelInputsChannelId = getChannelID("nxmgmt_label_inputs");
//        if(labelInputsChannelId == INVALID_CHANNEL_ID) {
//            LOG("ERROR: Invalid channelID for nxmgmt_mc_inputs \n");
//        }
//    }
//}

void applyInputs() {
//    if (mcInputsChannelId == INVALID_CHANNEL_ID) initMCInputsChannel();
    int biasArray[NUM_MCS];
//    int biasArray1[10];
//    LOG("reading MCAD input biases... \n");
//    LOG("Pixel group ID=%d \n", MCSOMA_CXGRP_ID);

//    LOG("NUM_MCS =%d \n", NUM_MCS);
    readChannel(mcInputsChannelId, &biasArray, NUM_MCS);

    for (int i = 0; i < NUM_MCS; i++) {
        int bias1 = biasArray[i];
        nxCompartmentGroup[MCSOMA_CXGRP_ID][i].Bias = bias1;
//        int ctmp = dtrite[i];
//        nxCompartment[ctmp].Bias = bias1;
//        bias1 = nxCompartmentGroup[MCSOMA_CXGRP_ID][i].Bias;
//        LOG("BIAS for input pixel [%d]=%d \n",i,bias1);
    }

}
//
void applyLabels() {
//    if (labelInputsChannelId == INVALID_CHANNEL_ID) initLabelChannel();

    int labelbiasArray[NUM_TARGETS];
//    for (int i = 0; i < 10; i++) {
//        int bias1 = nxCompartmentGroup[LABEL_CXGRP_ID][i].Bias;
//        LOG("BIAS for input label [%d]=%d \n",i,bias1);
//    }
//    LOG("reading Label input biases... \n");
//    LOG("Label group ID=%d \n", LABEL_CXGRP_ID);
    readChannel(labelInputsChannelId, &labelbiasArray, NUM_TARGETS);
    for (int i = 0; i < NUM_TARGETS; i++) {
        int bias1 = labelbiasArray[i];
        nxCompartmentGroup[LABEL_CXGRP_ID][i].Bias = bias1;
////        randarr[i] = 1;
//        if (labelbiasArray[i] > 0){
//            randarr[i] = 0;
////            printf("target %d ", i);
//        }
//        if (i ==0 || i ==1 || i == 6){
//            randarr[i] = 1;
////            printf("target %d ", i);
//        }
//        if (i ==0 || i ==4){
//            randarr[i] = 1;
////            printf("target %d ", i);
//        }

//        printf("target bit %d ", i);
//        printf("rand %d \n", randarr[i]);

//        bias = nxCompartmentGroup[LABEL_CXGRP_ID][i].Bias;
//        LOG("BIAS for input label [%d]=%d \n",i,bias1);
//        int dc = nxCompartmentGroup[LABEL_CXGRP_ID][i].Decay_v;
//        LOG("Decay_v [%d]=%d \n",i,dc);
    }

//    if (resetmode==0){
//        nxCompartmentGroup[LAST_ECLAYER_CXGRP_ID].Decay_v = 4095;
//        nxCompartmentGroup[LAST_ECLAYER_CXGRP_ID+1].Decay_v = 4095;
//        nxCompartmentGroup[LAST_ECLAYER_CXGRP_ID].Vth = 255;
//        nxCompartmentGroup[LAST_ECLAYER_CXGRP_ID+1].Vth = 255;
//        for (int i = 0; i < NUM_TARGETS; i++) {
//            if (randarr[i] == 1){
//                nxCompartmentGroup[LAST_ECLAYER_CXGRP_ID][i].Decay_v = 0;
//                nxCompartmentGroup[LAST_ECLAYER_CXGRP_ID+1][i].Decay_v = 0;
//                nxCompartmentGroup[LAST_ECLAYER_CXGRP_ID][i].Vth = 8;
//                nxCompartmentGroup[LAST_ECLAYER_CXGRP_ID+1][i].Vth = 8;
//
//            }
//        }
//
//    }

    if (resetmode==0){
//        time_t t;
//        srand((unsigned) time(&t));
        srand(time(0));
        int randarr[NUM_TARGETS];
        int droprate = 3;
        for (int j = 0; j < NUM_TARGETS; j++) {
//            randarr[j] = rand()%10;
            randarr[j] = 10;
            if (labelbiasArray[j] > 0){
                randarr[j] = 10;
//              printf("target %d ", i);
            }
//            if (j == 0){
//                randarr[j] = 10;
//            }
//            if (j == 1){
//                randarr[j] = 10;
//            }
//            if (j == 6){
//                randarr[j] = 10;
//            }
        }
//        nxCompartmentGroup[LAST_ECLAYER_CXGRP_ID].Decay_v = 4095;
//        nxCompartmentGroup[LAST_ECLAYER_CXGRP_ID+1].Decay_v = 4095;
//        nxCompartmentGroup[LAST_ECLAYER_CXGRP_ID].Vth = 3255;
//        nxCompartmentGroup[LAST_ECLAYER_CXGRP_ID+1].Vth = 3255;
        for (int k = 0; k < NUM_TARGETS; k++) {
            if (randarr[k] >= droprate){
                if (labelbiasArray[k] > 0){
                    nxCompartmentGroup[LAST_POSECLAYER_CXGRP_ID][k].Decay_v = 0;
                    nxCompartmentGroup[LAST_NEGECLAYER_CXGRP_ID][k].Decay_v = 0;
                    nxCompartmentGroup[LAST_POSECLAYER_CXGRP_ID][k].Vth = 8;
                    nxCompartmentGroup[LAST_NEGECLAYER_CXGRP_ID][k].Vth = 8;
                }
                else {
                    nxCompartmentGroup[LAST_POSECLAYER_CXGRP_ID][k].Decay_v = 0;
                    nxCompartmentGroup[LAST_NEGECLAYER_CXGRP_ID][k].Decay_v = 0;
                    nxCompartmentGroup[LAST_POSECLAYER_CXGRP_ID][k].Vth = 8;
                    nxCompartmentGroup[LAST_NEGECLAYER_CXGRP_ID][k].Vth = 8;
                }

            }
            else if (randarr[k] < droprate){
                nxCompartmentGroup[LAST_POSECLAYER_CXGRP_ID][k].Decay_v = 4095;
                nxCompartmentGroup[LAST_NEGECLAYER_CXGRP_ID][k].Decay_v = 4095;
                nxCompartmentGroup[LAST_POSECLAYER_CXGRP_ID][k].Vth = 3255;
                nxCompartmentGroup[LAST_NEGECLAYER_CXGRP_ID][k].Vth = 3255;
            }
        }

//        nxCompartmentGroup[LAST_ECLAYER_CXGRP_ID].Decay_v = 0;
//        nxCompartmentGroup[LAST_ECLAYER_CXGRP_ID+1].Decay_v = 0;
//        nxCompartmentGroup[LAST_ECLAYER_CXGRP_ID].Vth = 8;
//        nxCompartmentGroup[LAST_ECLAYER_CXGRP_ID+1].Vth = 8;
////        int dc = nxCompartmentGroup[LAST_ECLAYER_CXGRP_ID][0].Decay_v;
////        int dv = nxCompartmentGroup[LAST_ECLAYER_CXGRP_ID][0].Vth;
////        LOG("Decay_v [%d]=%d Vth %d\n",LAST_ECLAYER_CXGRP_ID,dc, dv);
    }
    else if (resetmode==1){
        nxCompartmentGroup[LAST_POSECLAYER_CXGRP_ID].Decay_v = 4095;
        nxCompartmentGroup[LAST_NEGECLAYER_CXGRP_ID].Decay_v = 4095;
        nxCompartmentGroup[LAST_POSECLAYER_CXGRP_ID].Vth = 3255;
        nxCompartmentGroup[LAST_NEGECLAYER_CXGRP_ID].Vth = 3255;
//        int dc = nxCompartmentGroup[LAST_ECLAYER_CXGRP_ID][0].Decay_v;
//        int dv = nxCompartmentGroup[LAST_ECLAYER_CXGRP_ID][0].Vth;
//        LOG("Decay_v [%d]=%d Vth %d\n",LAST_ECLAYER_CXGRP_ID,dc, dv);
    }
}


//void initMgmtChannel() {
//    if(mgmtChannelId == INVALID_CHANNEL_ID) {
//        mgmtChannelId = getChannelID("nxmgmt_input_axon_ids");
//        if(mgmtChannelId == INVALID_CHANNEL_ID)
//            LOG("ERROR: Invalid channelID for nxinit\n");
//    }
//}


void switchToPositiveThetaTrain() {
//    ResetAllCx(s);
//    thetaState = POSITIVE_THETA;
    applyInputs();
//    applyLabels();
//    tbeginCurrState = s->time_step;
}

void switchToPositiveThetaTest() {
//    ResetAllCx(s);
//    thetaState = POSITIVE_THETA;
    applyInputs();
//    applyLabels();
//    tbeginCurrState = s->time_step;
}

//int changeMode() {
//
//    /*if (mgmtChannelId == INVALID_CHANNEL_ID) initMgmtChannel();
//    readChannel(mgmtChannelId,&mode,1);
//    LOG("MODE = %d \n", mode);*/
//
//    switch(mode) {
//        case TRAINING:
//            mode = TESTING;
////            LOG("MODE = %d \n", mode);
//            return 0;
////            switchToPositiveThetaTest(s); break;
//        // just to supress compiler warnings; will not encounter this case
//        case TESTING:
//            mode = TRAINING;
////            LOG("MODE = %d \n", mode);
//            return 1;
////            switchToPositiveThetaTrain(s); break;
//        default:
//            break;
//    }
////    (s);
//
//}


//void ResetTraceandCxState(int coreId, int num_overwrite_compartments){
//
//    NeuronCore *nc = NEURON_PTR(nx_nth_coreid(coreId));
//    nx_fast_init32(nc->cx_meta_state, 1024, 0);
//    nx_fast_init32(nc->cx_state, 1024, 0);
//
//}

//void ResetTraceandCxState(int coreId, int num_overwrite_compartments, CxState cxs, MetaState ms){
//
//    NeuronCore *nc = NEURON_PTR(nx_nth_coreid(coreId));
//
//    nx_fast_init32(nc->cx_meta_state, 1024, *(uint32_t*)&ms);
//    nx_fast_init64(nc->cx_state, 1024, *(uint64_t*)&cxs);
////    nx_fast_init32(nc->dendrite_accum, 8192/2, 0);
//
//}
//
//void ResetTraces(int coreId, int compartment_id, int traceProfile, int stdpProfile){
//
//    NeuronCore *nc = NEURON_PTR(nx_nth_coreid(coreId));
////        printf("writing core %d \n", global_overwrite_core_ids[ii]);
////        printf("writing compartment %d \n", global_overwrite_compartment_ids[ii]);
////        nc->stdp_post_state[global_overwrite_compartment_ids[ii]] = global_post_trace_entry[ii];
//    nc->stdp_post_state[compartment_id] = (PostTraceEntry) {
//            .Yspike0      = 0,
//            .Yspike1      = 0,
//            .Yspike2      = 0,
//            .Yepoch0      = 0,
//            .Yepoch1      = 0,
//            .Yepoch2      = 0,
//            .Tspike       = 0,
//            .TraceProfile = traceProfile,
//            .StdpProfile  = stdpProfile
//        };
//
//}

void ResetAllCx(){

//    CxState cxs = (CxState) {.U=0, .V=0};
//    MetaState ms = (MetaState) {.Phase0=2, .SomaOp0=3,
//                                 .Phase1=2, .SomaOp1=3,
//                                 .Phase2=2, .SomaOp2=3,
//                                 .Phase3=2, .SomaOp3=3};

    NeuronCore *nc;
//    int tmp = 1;
//    if (tmp==1){
    if (resetmode==1){
//        if (resetmode==1){
//            for (int i = 1; i < CONV_GC_CORE_ID_BEGIN; i=i+1) {
//        //        LOG("Reset coreId=%d\n", i);
//    //            ResetTraceandCxState(i, 100, cxs, ms);
//    //            printf("writing core %d \n", i);
//    //            NeuronCore *nc;
//                nc = NEURON_PTR(nx_nth_coreid(i));
//
//    //            nx_fast_init(nc->synapse_map[1], 4096, 4, 8, (uint32_t[1]) {0});
//    //            nx_fast_init32(nc->cx_meta_state, 1024, 0);
//    //            nx_fast_init64(nc->cx_state, 1024, 0);
//    //            nx_fast_init32(nc->dendrite_accum, 8192/2, 0);
//
//    //            nx_fast_init(&nc->synapse_map[1], 2032, 4, 8, (uint32_t[1]) {0});
//    //            nx_fast_init32(nc->cx_meta_state, 1024, 0);
//                nx_fast_init64(nc->cx_state, 1024, 0);
//    //            nx_fast_init32(nc->dendrite_accum, 8192/2, 0);
//                nx_flush_core(nx_nth_coreid(i));
//    //
//    //            nx_fast_init(&neuron->synapse_map[1], numAxon, 4, 8, (uint32_t[1]) {0});
//    //            nx_fast_init32(&neuron->cx_meta_state, numCx, 0);
//    //            nx_fast_init32(&neuron->cx_state, 2 * numCx, 0);
//            }
//        }

        for (int i = CONV_GC_CORE_ID_BEGIN; i < GC_CORE_ID_BEGIN; i=i+1) {
    //        LOG("Reset coreId=%d\n", i);
//            ResetTraceandCxState(i, 100, cxs, ms);
//            printf("writing core %d \n", i);
//            NeuronCore *nc;
            nc = NEURON_PTR(nx_nth_coreid(i));

//            nx_fast_init(nc->synapse_map[1], 4096, 4, 8, (uint32_t[1]) {0});
//            nx_fast_init32(nc->cx_meta_state, 1024, 0);
//            nx_fast_init64(nc->cx_state, 1024, 0);
//            nx_fast_init32(nc->dendrite_accum, 8192/2, 0);

//            nx_fast_init(&nc->synapse_map[1], 2032, 4, 8, (uint32_t[1]) {0});
            nx_fast_init32(nc->cx_meta_state, 1024, 0);
            nx_fast_init64(nc->cx_state, 1024, 0);
            nx_fast_init32(nc->dendrite_accum, 8192/2, 0);
//            nx_flush_core(nx_nth_coreid(i));

//
//            nx_fast_init(&neuron->synapse_map[1], numAxon, 4, 8, (uint32_t[1]) {0});
//            nx_fast_init32(&neuron->cx_meta_state, numCx, 0);
//            nx_fast_init32(&neuron->cx_state, 2 * numCx, 0);
        }

//        for (int i = GC_CORE_ID_BEGIN; i <= GC_CORE_ID_END; i=i+2) {
        for (int i = 0; i < NUM_GC_CORES; i=i+1) {
    //        LOG("Reset coreId=%d\n", i);
//            ResetTraceandCxState(i, 100, cxs, ms);
//            printf("writing core %d \n", global_overwrite_cores[i]);
//            NeuronCore *nc;

//            nc = NEURON_PTR(nx_nth_coreid(i));
            nc = NEURON_PTR(nx_nth_coreid(global_overwrite_cores[i]));

//            nx_fast_init(&nc->synapse_map[1], 4096, 4, 8, (uint32_t[1]) {0});
//            nx_fast_init32(&nc->synapse_map[1], 4096, 0);

            nx_fast_init32(nc->cx_meta_state, 1024, 0);
            nx_fast_init64(nc->cx_state, 1024, 0);
            nx_fast_init32(nc->dendrite_accum, 8192/2, 0);
//            nx_flush_core(nx_nth_coreid(global_overwrite_cores[i]));
        }
    }


    for(int ii=0; ii<NUM_GCS; ii++)
    {

//        ResetTraces(global_overwrite_core_ids[ii], global_overwrite_compartment_ids[ii], traceProfile[ii], stdpProfile[ii]);

        nc = NEURON_PTR(nx_nth_coreid(global_overwrite_core_ids[ii]));
//        printf("writing core %d \n", global_overwrite_core_ids[ii]);
//        printf("writing compartment %d \n", global_overwrite_compartment_ids[ii]);
//        nc->stdp_post_state[global_overwrite_compartment_ids[ii]] = global_post_trace_entry[ii];
        nc->stdp_post_state[global_overwrite_compartment_ids[ii]] = (PostTraceEntry) {
            .Yspike0      = 0,
            .Yspike1      = 0,
            .Yspike2      = 0,
            .Yepoch0      = 0,
            .Yepoch1      = 0,
            .Yepoch2      = 0,
            .Tspike       = 0,
            .TraceProfile = traceProfile[ii],
            .StdpProfile  = stdpProfile[ii]
        };
        //printf("Item %d writing core %d STDP_POST_STATE register %d y1 trace with value %d\n", ii, global_overwrite_core_ids[ii], global_overwrite_compartment_ids[ii], global_post_trace_entry[ii].Yepoch0);
    }

}


void runMgmt(runState *s) {
//    LOG("\n MGMT %d: BEGIN command=%s \n", s->time_step, command2strings[command]);
    switch(command) {
        case RESET:
            ResetAllCx();
            applyLabels();
            break;
        case SWITCH_TO_POSITIVE_THETA_TRAINING:
            switchToPositiveThetaTrain(); break;
        case SWITCH_TO_POSITIVE_THETA_TESTING:
            switchToPositiveThetaTest(); break;
        case DO_NOTHING:
            break;
        default:
            break;
    }

//    if (USE_LMT_SPIKE_COUNTERS) updateSpikeCounters(s->time_step);

//    LOG("\n MGMT %d: END command=%s \n", s->time_step, command2strings[command]);
}
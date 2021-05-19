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

#include "initsnip.h"


//int global_overwrite_core_ids[NUM_GCS];
//int global_overwrite_compartment_ids[NUM_GCS]; //stdp_post_state is indexed by compartmentId
//PostTraceEntry global_post_trace_entry[NUM_GCS];

static int channelID = INVALID_CHANNEL_ID;
//static int CoreIdsChannelId = INVALID_CHANNEL_ID;
//static int StdpCompartmentIndexChannelId = INVALID_CHANNEL_ID;
//static int StdpProfileChannelId = INVALID_CHANNEL_ID;
//static int TraceProfileChannelId = INVALID_CHANNEL_ID;

void initChannel() {
    if(channelID == INVALID_CHANNEL_ID) {
        channelID = getChannelID("nxinit");
        if(channelID == INVALID_CHANNEL_ID) {
//          LOG("ERROR: Invalid channelID for nxinit\n");
        }
    }
//    if(CoreIdsChannelId == INVALID_CHANNEL_ID) {
//        CoreIdsChannelId = getChannelID("nxinitCoreIds");
//        if(CoreIdsChannelId == INVALID_CHANNEL_ID) {
////          LOG("ERROR: Invalid channelID for nxinitCoreIds\n");
//        }
//    }
//    if(StdpCompartmentIndexChannelId == INVALID_CHANNEL_ID) {
//        StdpCompartmentIndexChannelId = getChannelID("nxinitStdpCompartmentIndex");
//        if(StdpCompartmentIndexChannelId == INVALID_CHANNEL_ID) {
////          LOG("ERROR: Invalid channelID for nxinitStdpCompartmentIndex\n");
//        }
//    }
//    if(StdpProfileChannelId == INVALID_CHANNEL_ID) {
//        StdpProfileChannelId = getChannelID("nxinitStdpProfile");
//        if(StdpProfileChannelId == INVALID_CHANNEL_ID) {
////          LOG("ERROR: Invalid channelID for nxinitStdpProfile\n");
//        }
//    }
//    if(TraceProfileChannelId == INVALID_CHANNEL_ID) {
//        TraceProfileChannelId = getChannelID("nxinitTraceProfile");
//        if(TraceProfileChannelId == INVALID_CHANNEL_ID) {
////          LOG("ERROR: Invalid channelID for nxinitTraceProfile\n");
//        }
//    }
}
//
//
void readInputs() {

////    LOG("NUM_GCS %d \n", NUM_GCS);
//
////    //which coreIds do our compartments lie on?
////    channelID = getChannelID("nxinitCoreIds");
////    if(channelID == -1) {
////      printf("Invalid channelID for nxinitCoreIds\n");
////    }
////    LOG("CoreIdsChannelId\n");
//    readChannel(CoreIdsChannelId,&global_overwrite_core_ids,NUM_GCS);
//
//
////    //which stdp state registers are we modifying
////    channelID = getChannelID("nxinitStdpCompartmentIndex");
////    if(channelID == -1) {
////      printf("Invalid channelID for nxinitStdpCompartmentIndex\n");
////    }
////    LOG("StdpCompartmentIndexChannelId\n");
//    readChannel(StdpCompartmentIndexChannelId,&global_overwrite_compartment_ids,NUM_GCS);
//
//
//    //----------------what are the stdp post trace states states at initialization?
//    // (since we cannot read them while the chip is running)
//
//    // stdp profile
//    int stdpProfile[NUM_GCS];
////    channelID = getChannelID("nxinitStdpProfile");
////    if(channelID == -1) {
////      printf("Invalid channelID for nxinitStdpProfile\n");
////    }
////    LOG("StdpProfileChannelId\n");
//    readChannel(StdpProfileChannelId,&stdpProfile,NUM_GCS);
//
//    // trace profile
//    int traceProfile[NUM_GCS];
////    channelID = getChannelID("nxinitTraceProfile");
////    if(channelID == -1) {
////      printf("Invalid channelID for nxinitTraceProfile\n");
////    }
////    LOG("TraceProfileChannelId\n");
//    readChannel(TraceProfileChannelId,&traceProfile,NUM_GCS);
//
//        // explicitly initialize the values for all fields. Probably unnecessary, but I'm not sure what they do?
//    // TraceProfile presumably determines trace behaviour (decay/impulse) but we'll overwrite it at every timestep anyway, so shouldn't matter
//    // stdpProfile may be involved in determining which stdp rule to use, so we should make sure it is correct
//    // Tspike probably indicates when a spike arrived? Doesn't matter since we overwrite traces
//    // epoch or spike probably encode the trace
//    for(int ii=0; ii<NUM_GCS; ii++)
//    {
//        global_post_trace_entry[ii].Yspike0=0;
//        global_post_trace_entry[ii].Yspike1=0;
//        global_post_trace_entry[ii].Yspike2=0;
//        global_post_trace_entry[ii].Yepoch0=0;
//        global_post_trace_entry[ii].Yepoch1=0;
//        global_post_trace_entry[ii].Yepoch2=0;
//        global_post_trace_entry[ii].Tspike=0;
//        global_post_trace_entry[ii].TraceProfile=traceProfile[ii];
//        global_post_trace_entry[ii].StdpProfile=stdpProfile[ii];
//    }

    int inputData[NUM_MCS+NUM_TARGETS];
    readChannel(channelID, &inputData, NUM_MCS+NUM_TARGETS);
//    LOG("reading MCAD input biases... \n");
//    LOG("Pixel group ID=%d \n", MCSOMA_CXGRP_ID);
//    LOG("Label group ID=%d \n", LABEL_CXGRP_ID);
//    LOG("NUM_MCS =%d \n", NUM_MCS);
    for (int i = 0; i < NUM_MCS; i++) {
        int bias1 = inputData[i];
        nxCompartmentGroup[MCSOMA_CXGRP_ID][i].Bias = bias1;
//        int ctmp = dtrite[i];
//        nxCompartment[ctmp].Bias = bias1;
//        LOG("BIAS for Pixels[%d]=%d \n",i,bias1);
    }
    for (int i = NUM_MCS; i < NUM_MCS+NUM_TARGETS; i++) {
        int bias1 = inputData[i];
        nxCompartmentGroup[LABEL_CXGRP_ID][i-NUM_MCS].Bias = bias1;
//        LOG("BIAS for Labels[%d]=%d \n",(i-NUM_MCS),bias1);
    }
}
//
//void dumpConstants() {
//    LOG("NUM_CORES = %d \n", NUM_CORES);
//    LOG("NUM_MCS = %d \n", NUM_MCS);
//    LOG("NUM_GCS = %d \n", NUM_GCS);
//    LOG("MCSOMA_CXGRP_ID = %d \n", MCSOMA_CXGRP_ID);
//}

void initParamsAndInputs(runState *s) {
    //dumpConstants();
    initChannel();
    readInputs();
//    thetaState = POSITIVE_THETA;
    if (TRAIN == 1){
    mode = TRAINING;
    }
    if (TRAIN == 0){
    mode = TESTING;
    }

//    if (mode == TRAINING) LOG("INIT: TRAINING MODE \n");
//    tbeginCurrState = s->time_step;
//    LOG("DONE INIT: tstep %d \n", tbeginCurrState);
}

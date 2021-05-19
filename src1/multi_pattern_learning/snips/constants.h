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

#ifndef CONSTANTS_H
#define CONSTANTS_H 

#define INVALID_CHANNEL_ID -1
#define TRAIN 1
#define NUM_PATTERNS 2
#define NUM_CORES 56
#define NUM_MCS 784
#define NUM_GCS 110
#define NUM_TARGETS 10
#define MCAD_CXGRP_ID 
#define MCSOMA_CXGRP_ID 0
#define LABEL_CXGRP_ID 1
#define LAST_ECLAYER_CXGRP_ID 14
#define GAMMA_CYCLE_DURATION 64
#define NO_LEARNING_PERIOD 20
#define CONV_GC_CORE_ID_BEGIN 2
#define GC_CORE_ID_BEGIN 14
#define GC_CORE_ID_END 55
#define LABEL_CORE_ID 56
#define NUM_TRAIN_SAMPLES 60000
#define NUM_TEST_SAMPLES 0
#define USE_LMT_SPIKE_COUNTERS 0
#define RUN_TIME 7680000

static const int gcGrpIdsPerPattern[NUM_PATTERNS] = {6,13,};

static const int NUM_GCS_PER_PATTERN[NUM_PATTERNS] = {100,10,};

#endif
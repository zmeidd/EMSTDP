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

#ifndef INITSNIP_H
#define INITSNIP_H
#include "nxsdk.h"
#include "constants.h"
#include "common.h"

//#ifndef INITIALIZATION_H
//#define INITIALIZATION_H
//
//#include "nxsdk.h"
//#include "array_sizes.h"

//extern int global_overwrite_core_ids[NUM_GCS];
//extern int global_overwrite_compartment_ids[NUM_GCS];
//extern PostTraceEntry global_post_trace_entry[NUM_GCS];
//void initChannel();
//void readEplParams(int *param);
//void readParams();
//bool validateParams();
//void readInputs();
//void dumpConstants();
void initParamsAndInputs(runState *s);
//int doInit(runState *s);
#endif

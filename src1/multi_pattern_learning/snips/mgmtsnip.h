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

#ifndef MGMTSNIP_H
#define MGMTSNIP_H
#include "nxsdk.h"
#include "constants.h"
#include "common.h"
#include "spu.h"
#include "initsnip.h"

int doMgmt(runState *s);
void runMgmt(runState *s);
//void disableLearning();
//void switchToPositiveThetaTest(runState *s);
//void switchToNegativeTheta(runState *s);
//void changeMode(runState *s);
//void continueTraining(runState *s);
//void switchToInference(runState *s);
//void changeMCToGCWeights(int coreId);
//void changeMCToGCConnections();
//void changeGCToMCExcConnections();
//void changeGCToMCWeights(int coreId);
//void initMgmtChannel();
//void initMCInputsChannel();
#endif

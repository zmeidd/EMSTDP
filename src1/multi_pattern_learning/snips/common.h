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

#ifndef COMMON_H
#define COMMON_H

//bool doThetaReset;
//int tbeginCurrState;
//enum ThetaState {POSITIVE_THETA, NEGATIVE_THETA} thetaState;
enum Mode {TRAINING, TESTING} mode;
enum Command {DO_NOTHING, RESET,
                SWITCH_TO_POSITIVE_THETA_TESTING,
                SWITCH_TO_POSITIVE_THETA_TRAINING} command;

//static const char *command2strings[] = {"DO_NOTHING", "RESET",
//            "SWITCH_TO_POSITIVE_THETA_TESTING",
//            "SWITCH_TO_POSITIVE_THETA_TRAINING"};

#endif
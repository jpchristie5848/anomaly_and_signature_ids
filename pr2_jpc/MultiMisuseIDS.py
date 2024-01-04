import os
import random
import math

from copy import deepcopy
from src.Preprocessing import convert_to_arff, data_to_dict, LAST_ATTR
from datetime import datetime

#
# PRE-PROCESSING SECTION START
#

# the attack types to create IDS's for
# includes Optimized_Normal for anomaly based IDS
ids_attack_types = ["Optimized_Neptune", "Optimized_Back", "Optimized_Smurf", "Optimized_Satan",
                    "Optimized_PortSweep"]

# used to map types to their corresponding output nodes
ids_attack_types_map = {0: "Optimized_Neptune", 1: "Optimized_Back", 2: "Optimized_Smurf", 3: "Optimized_Satan",
                        4: "Optimized_PortSweep", 5: "other"}

file_names = os.listdir()  # get the names of all the files in this directory

attack_types = []
for name in file_names:
    if "Optimized" in name:
        attack_types.append(name)

data_files = data_to_dict(attack_types)  # create dict containing data for all attack types

# create lists for training and testing data
train_data = []
test_data = []

train_record_counts = {}
test_record_counts = {}

for i in range(len(attack_types)):
    train_record_counts[attack_types[i]] = 0
    test_record_counts[attack_types[i]] = 0

target_min = 20  # minimum number of each type of instance allowed in the data

train_done = False
test_done = False

for attack_type in attack_types:
    line_no = 0
    for rand, line in data_files[attack_type]:
        if ',' in line:
            if line_no % 3 != 0:  # only train on 2/3 of data
                train_data.append((rand, line))

                train_record_counts[attack_type] += 1  # increment record count for the attack type of the recorded
                                                       # instance

                train_done = True
                if train_record_counts[attack_type] < target_min:
                    train_done = False

                if train_done and test_done:
                    break

            else:  # only test on 1/3 of data
                test_data.append((rand, line))

                attack_type = line.split(',')[LAST_ATTR].strip()
                test_record_counts[attack_type] += 1  # increment record count for the attack type of the recorded
                                                      # instance

                test_done = True
                if test_record_counts[attack_type] < target_min:
                    test_done = False

                if train_done and test_done:
                    break
        line_no += 1

train_data.sort()
test_data.sort()

# create arff file with training data
convert_to_arff("multi_misuse_train", train_data, attack_types)
# create arff file with testing data
convert_to_arff("multi_misuse_test", test_data, attack_types)

# PRE-PROCESSING SECTION END
#

#
# TRAINING SECTION START
#

class Neural_Net():

    NUM_HIDDEN_NODES = 10
    NUM_ATTR = 41
    NUM_OUTPUT_NODES = 6  # 5 attacks + other

    def __init__(self):

        self.output = []

        self.input_layer_weights = []
        self.hidden_layer_weights = []

        self.hidden_layer_outputs = []
        self.hidden_layer_inputs = []

        for x in range(self.NUM_ATTR):
            self.input_layer_weights.append([])
            for y in range(self.NUM_HIDDEN_NODES):
                self.input_layer_weights[x].append(random.random())

        for x in range(self.NUM_HIDDEN_NODES):
            self.hidden_layer_weights.append([])
            for y in range(self.NUM_OUTPUT_NODES):
                self.hidden_layer_weights[x].append(random.random())

    def feed_forward(self, instance):

        inputs = instance.split(',')[:-1]  # get attr values for the instance (remove last one because that's the label)
        self.hidden_layer_outputs = []  # unweighted values coming out of hidden layer nodes

        # for each hidden node, calculate the weighted sum of its inputs and apply an activation function to it
        for i in range(self.NUM_HIDDEN_NODES):
            weighted_sum = 0
            for j in range(self.NUM_ATTR):
                weighted_sum += float(inputs[j]) * self.input_layer_weights[j][i]  # multiply input values by weights
                                                                                   # and add it to weighted sum
            self.hidden_layer_inputs.append(weighted_sum)

            self.hidden_layer_outputs.append(max(0, weighted_sum))  # apply activation function (ReLU)

        # for each output node, calculate the weighted sum of its inputs and apply and activation function to it
        for i in range(self.NUM_OUTPUT_NODES):
            weighted_sum = 0
            for j in range(self.NUM_HIDDEN_NODES):
                weighted_sum += self.hidden_layer_outputs[j] * self.hidden_layer_weights[j][i]
            self.output.append(max(0, weighted_sum))

    def calculate_dc_dw(self, instance):

        inputs = instance.split(',')[:-1]  # get attr values for the instance (remove last one because that's the label)

        label = instance.split(',')[LAST_ATTR].strip()  #

        # calculate d(C)/d(w) for each weight

        first_layer = deepcopy(self.input_layer_weights)
        second_layer = deepcopy(self.hidden_layer_weights)

        # do first layer of weights
        for i in range(self.NUM_ATTR):
            for j in range(self.NUM_HIDDEN_NODES):

                # 3 terms needed:

                # first term: d(C)/d(a), the derivative of cost with respect to the activation output of the node the
                # weight points to.

                # equal to the sum from k=0 to n-1 of d(C)/d(a(k)) * d(a)/d(z(k)) * d(z)/d(a(j))
                # where n is the number of output nodes

                First = 0

                for x in range(self.NUM_OUTPUT_NODES):

                    # first term: d(C)/d(a(k)), the derivative of cost with respect to the activation output of the output
                    # node

                    if label == ids_attack_types_map[x]:  # find the optimal value of the output given the current label.
                        optimal = 1  # 1 if the label is the attack type being detected, 0 if not
                    else:  #
                        if ids_attack_types_map[x] == "other" and label not in ids_attack_types:
                            optimal = 1
                        else:
                            optimal = 0  # For single misuse IDS

                    first = 2 * (self.output[x] - optimal)  # derivative of loss function

                    # second term: d(a)/d(z(k)), the derivative of the activation output of the node the weight points to
                    # with respect to the input of the node the weight points to

                    if self.output[x] > 0:  #
                        dReLU = 1  # if z > 0 the derivative of ReLU is 1,
                    else:  # otherwise it is 0
                        dReLU = 0  #

                    second = dReLU

                    # third term: d(z(k))/d(a(j))
                    # equal to the weight that points from the node the current weight points to output node k

                    third = self.hidden_layer_weights[j][x]

                    First += first * second * third

                # second term: d(a)/d(z), the derivative of the activation output of the node the weight points to with
                # respect to the input of the node the weight points to

                if self.hidden_layer_inputs[j] > 0:  #
                    dReLU = 1  # if z > 0 the derivative of ReLU is 1,
                else:  # otherwise it is 0
                    dReLU = 0  #

                Second = dReLU

                # third term: d(z)/d(w), the derivative of the input of the node the weight points to with respect to
                # the weight

                Third = float(inputs[i])

                result = First * Second * Third

                first_layer[i][j] = result

        # do second layer of weights
        for i in range(self.NUM_HIDDEN_NODES):
            for x in range(self.NUM_OUTPUT_NODES):
                # 3 terms needed:

                # first term: d(C)/d(a), the derivative of cost with respect to the activation output of the node the
                # weight points to

                if label == ids_attack_types_map[x]:  # find the optimal value of the output given the current label.
                    optimal = 1  # 1 if the label is the attack type being detected, 0 if not
                else:  #
                    if ids_attack_types_map[x] == "other" and label not in ids_attack_types:
                        optimal = 1
                    else:
                        optimal = 0  # For single misuse IDS

                first = 2 * (self.output[x] - optimal)  # derivative of loss function

                # second term: d(a)/d(z), the derivative of the activation output of the node the weight points to with
                # respect to the input of the node the weight points to

                if self.output[x] > 0:  #
                    dReLU = 1  # if z > 0 the derivative of ReLU is 1,
                else:  # otherwise it is 0
                    dReLU = 0  #

                second = dReLU

                # third term: d(z)/d(w), the derivative of the input of the node the weight points to with respect to the
                # weight

                third = self.hidden_layer_outputs[i]

                result = first * second * third
                second_layer[i][x] = result

        return [first_layer, second_layer]

    def test(self, test_lines):

        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for line in test_lines:
            if ',' in line and '@' not in line:
                self.feed_forward(line)

                label = line.split(',')[LAST_ATTR].strip()

                output_dists = []
                for x in range(len(self.output)):
                    output_dists.append(math.fabs(1 - self.output[x]))

                self.output = []

                m = output_dists.index(min(output_dists))

                classification = ids_attack_types_map[m]

                if label == classification:
                    TP += 1
                elif label != classification:
                    if label not in ids_attack_types and classification == "other":
                        TN += 1
                    elif label in ids_attack_types:
                        FN += 1
                    else:
                        FP += 1

        return TP, FP, TN, FN

def run_IDS():
    learning_rate = 0.1

    TP = 0
    FP = 0
    FN = 0
    TN = 0

    TP_rate = 0
    FP_rate = 0
    TN_rate = 0
    FN_rate = 0

    total_epochs = 0

    NN_model = Neural_Net()

    with open("arff/multi_misuse_train_data.arff") as file:
        train_lines = file.readlines()
        file.close()
    with open("arff/multi_misuse_test_data.arff") as file:
        test_lines = file.readlines()
        file.close()

    print("learning rate:", learning_rate)

    start_time = datetime.now()
    while True:

        total_lines = 0

        dc_dw = []  # d(C)/d(w) for all weights in the model for each instance passed through

        for line in train_lines:
            if ',' in line and '@' not in line:
                NN_model.feed_forward(line)
                dc_dw.append(NN_model.calculate_dc_dw(line))  # for each instance, calculate loss with feed
                # forward and then calculate d(C)/d(w) for all
                # weights
                NN_model.output = []
                total_lines += 1

        # average d(C)/d(w) for each weight over all training samples
        dc_dw_avg = deepcopy(dc_dw[0])  # copy d(C)/d(w) array to calculate averages

        for y in range(total_lines):  #
            for i in range(NN_model.NUM_ATTR):  #
                for j in range(NN_model.NUM_HIDDEN_NODES):  # initialize all
                    dc_dw_avg[0][i][j] = 0.0  # d(C)/d(w) averages
            for i in range(NN_model.NUM_HIDDEN_NODES):  # to 0
                for x in range(NN_model.NUM_OUTPUT_NODES):
                    dc_dw_avg[1][i][x] = 0.0  #

        for y in range(total_lines):
            for i in range(NN_model.NUM_ATTR):  #
                for j in range(NN_model.NUM_HIDDEN_NODES):  # add up d(C)/d(w) totals
                    dc_dw_avg[0][i][j] += dc_dw[y][0][i][j]  # for each weight                        #
            for i in range(NN_model.NUM_HIDDEN_NODES):  #
                for x in range(NN_model.NUM_OUTPUT_NODES):
                    dc_dw_avg[1][i][x] += dc_dw[y][1][i][x]  #

        for i in range(NN_model.NUM_ATTR):  #
            for j in range(NN_model.NUM_HIDDEN_NODES):  # divide d(C)/d(w) totals by
                dc_dw_avg[0][i][j] /= float(total_lines)  # total instances to get the
        for i in range(NN_model.NUM_HIDDEN_NODES):  # averages
            for x in range(NN_model.NUM_OUTPUT_NODES):
                dc_dw_avg[1][i][x] /= float(total_lines)  #

        for i in range(NN_model.NUM_ATTR):
            for j in range(NN_model.NUM_HIDDEN_NODES):
                NN_model.input_layer_weights[i][j] -= dc_dw_avg[0][i][j] * \
                                                      learning_rate
        for i in range(NN_model.NUM_HIDDEN_NODES):
            for x in range(NN_model.NUM_OUTPUT_NODES):
                NN_model.hidden_layer_weights[i][x] -= dc_dw_avg[1][i][x] * \
                                                       learning_rate

        TP, FP, TN, FN = NN_model.test(test_lines)

        TP_rate = float(TP) / (float(TP) + float(FN))
        FP_rate = float(FP) / (float(FP) + float(TN))
        TN_rate = float(TN) / (float(FP) + float(TN))
        FN_rate = float(FN) / (float(TP) + float(FN))

        # print("Epoch", total_epochs + 1)
        # print("\tTP rate:", TP_rate, "| TN rate:", TN_rate, "| Correct:", (float(TP+TN)/float(TP+TN+FP+FN))*100, "%")

        total_epochs += 1

        if TP_rate >= 0.75 and TN_rate >= 0.75:  # desired performance threshold
            break
    #
    # TESTING SECTION START
    #

    print("Final Results:")
    print("\tTotal Epochs:", total_epochs)
    print("\tTrue Positive Rate:", TP_rate, str(TP) + "/" + str(TP + FN))
    print("\tFalse Positive Rate:", FP_rate, str(FP) + "/" + str(FP + TN))
    print("\tTrue Negative Rate:", TN_rate, str(TN) + "/" + str(FP + TN))
    print("\tFalse Negative Rate:", FN_rate, str(FN) + "/" + str(TP + FN))
    print("\tCorrectly Classified:", (float(TP + TN) / float(TP + TN + FP + FN)) * 100, "percent")
    print("\tIncorrectly Classified:", (float(FP + FN) / float(TP + TN + FP + FN)) * 100, "percent")
    end_time = datetime.now()
    diff = end_time - start_time
    elapsed_time = int((diff.seconds * 1000) + (diff.microseconds / 1000))
    print("Total time:", elapsed_time, "ms")


run_IDS()

import random

LAST_ATTR = 41  # the index of the last attribute

'''
Gets data from the specified files, shuffles it, and puts it into a dictionary
@param attack_types - the attack data files
@return the dictionary containing the data from these files
'''
def data_to_dict(attack_types):
    # Get data from files and put it into a dictionary
    data_files = {}
    for name in attack_types:
        with open(name, 'r') as file:
            lines = [(random.random(), line[:-1] + ", " + name + "\n") for line in file if ',' in line]
            lines.sort()
            data_files[name] = lines
            file.close()
    return data_files

'''
Writes attack data to an ARFF file
@param segment - Which segment this data is for. Can be "anomaly_train", "anomaly_test", "single_misuse_train", 
"single_misuse_test", "multi_misuse_train", or "multi_misuse_test"
@param data - the list containing the data to be written
@param attack_types - the attack types to be contained in the file
@param target_types - the attack types that are being classified by the IDS
@param target_min - the minimum number of records allowed for each target type
'''
def convert_to_arff(segment, data, attack_types):

    file = open('arff/'+segment+'_data.arff', 'w')
    file.write("@RELATION "+segment.upper()+"_DATA\n")

    for x in range(LAST_ATTR):
        file.write("@ATTRIBUTE ATTR" + str(x + 1) + " numeric\n")

    file.write("@ATTRIBUTE ATTACK_TYPE {")

    for x in range(len(attack_types)):
        if x == 0:
            file.write(attack_types[x])
        else:
            file.write(", " + attack_types[x])

    file.write("}\n")
    file.write("@DATA\n")

    for _, line in data:
        file.write(line)


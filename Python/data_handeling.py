import csv
import numpy as np

#Not tested yet

#Imports data from file.
#type can be 'categorical', 'numerical', 'exposure' or 'claims'
#
#Sizes of categorical data and numerical data are hard-coded for this
#special case (numerical: 3, categorical: 6 categories with levels 6, 3, 7, 11, 6, 21)

def import_data(filename, type):
    if type == 'categorical':
        dataOut = []
        with open(filename, encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            first_row = True
            for row in csv_reader:
                if first_row == True:
                    first_row = False
                else:
                    newSample = []
                    #vehPower-------------------------
                    newCat = list(np.zeros(6))
                    newCat[int(row[1])] = 1
                    newSample = newSample + newCat
                    #vehAge---------------------------
                    newCat = list(np.zeros(3))
                    newCat[int(row[2])] = 1
                    newSample = newSample + newCat
                    #drivAge--------------------------
                    newCat = list(np.zeros(7))
                    newCat[int(row[3])] = 1
                    newSample = newSample + newCat
                    #vehBrand-------------------------
                    newCat = list(np.zeros(11))
                    newCat[int(row[4])] = 1
                    newSample = newSample + newCat
                    #Area-----------------------------
                    newCat = list(np.zeros(6))
                    newCat[int(row[5])] = 1
                    newSample = newSample + newCat
                    #Region---------------------------
                    newCat = list(np.zeros(21))
                    newCat[int(row[6])] = 1
                    newSample = newSample + newCat
                    #---------------------------------
                    dataOut.append(newSample)
        return dataOut
    elif type == 'numerical':
        dataOut = []
        with open(filename, encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            first_row = True
            for row in csv_reader:
                if first_row == True:
                    first_row = False
                else:
                    newNum = []
                    newNum.append(float(row[0]))
                    newNum.append(float(row[1]))
                    newNum.append(float(row[2]))
                    dataOut.append(newNum)
        return dataOut
    elif type == 'exposure':
        dataOut = []
        with open(filename, encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            first_row = True
            for row in csv_reader:
                if first_row == True:
                    first_row = False
                else:
                    newSample = []
                    newSample.append(float(row[0]))
                    dataOut.append(newSample)
        return dataOut
    elif type == 'claims':
        dataOut = []
        with open(filename, encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            first_row = True
            for row in csv_reader:
                if first_row == True:
                    first_row = False
                else:
                    newSample = []
                    newSample.append(float(row[0]))
                    dataOut.append(newSample)
        return dataOut
    else:
        print('Import data type not specified - no data imported.')
        return 'Import data type not specified - no data imported.'





def get_one_category(data, category):
    dataOut = []

    #levels of categories are 6, 3, 7, 11, 6, 21
    if category == 1:
        start_index = 0
        end_index = 6
    elif category == 2:
        start_index = 6
        end_index = 6+3
    elif category == 3:
        start_index = 6+3
        end_index = 6+3+7
    elif category == 4:
        start_index = 6+3+7
        end_index = 6+3+7+11
    elif category == 5:
        start_index = 6+3+7+11
        end_index = 6+3+7+11+6
    elif category == 6:
        start_index = 6+3+7+11+6
        end_index = 6+3+7+11+6+21

    for i in range(len(data)):
        dataOut.append(data[i][start_index:end_index])

    return dataOut

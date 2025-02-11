#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "nnAuxillary.h"

double setToRandom(void){
    double output;
    output = 2*rand()/(RAND_MAX*1.0) - 1;
    return(output);
}

double setToZero(void){
    return(0.0);
}

double setToOne(void){
    return(1.0);
}


double actLinear(double input){
    return(input);
}

double actLinearPrim(double input){
    return(1.0);
}

double actSigmoid(double input){
    double output;
    output = 1/(1+exp(-input));
    return(output);
}

double actSigmoidPrim(double input){
    double output;
    output = actSigmoid(input)*(1 - actSigmoid(input));
    return(output);
}

double actTanh(double input){
    double output;
    double a;
    double b;

    a = exp(input);
    b = exp(-input);
    output = (a-b)/(a+b);

    return(output);
}


double actTanhPrim(double input){
    double output;
    double a;

    a = actTanh(input);
    output = 1 - a*a;

    return(output);
}
void getArray(double patterns[], double outputPattern[], int patternID, int patternSize){
    int i;
    for (i = 0; i < patternSize; i++){
        outputPattern[i] = patterns[patternID*patternSize + i];
    }
}

void generateOrder(int indexes[], int nIndex){
    int i;
    int j = 0;
    int okay;
    int newIndex;
    int largeRand;

    for (i = 0; i < nIndex; i++){
        indexes[i] = -1;
    }

    while(j < nIndex){
        largeRand = rand() + RAND_MAX*rand();
        newIndex = largeRand%nIndex;
        okay = 1;
        for (i = 0; i < j+1; i++){
            if (newIndex == indexes[i]){
                okay = 0;
                break;
            }
        }
        if (okay > 0){
            indexes[j] = newIndex;
            j = j + 1;
        }
    }
}

int arrMax_int(int *arr, int arrLength){
    int maxYet;
    int i;

    maxYet = arr[0];

    for (i = 0; i < arrLength; i++){
         if (arr[i] > maxYet){
             maxYet = arr[i];
         }
    }
    return(maxYet);
}

int maxIndex(double *arr, int arrLength){
    int index;
    int i;
    double maxYet;

    maxYet = arr[0];
    index = 0;

    for (i = 0; i < arrLength; i++){
         if (arr[i] > maxYet){
             maxYet = arr[i];
             index = i;
         }
    }
    return(index);
}

int maxIndexBetween(double *arr, int arrLength, int smallest, int largest){
    int index;
    int i;
    double maxYet;

    maxYet = arr[smallest];
    index = smallest;

    for (i = 0; i < arrLength; i++){
         if ((i > (smallest - 1))&&(i < (largest + 1))){
             if (arr[i] > maxYet){
                 maxYet = arr[i];
                 index = i;
             }
         }
    }
    return(index);
}

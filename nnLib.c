#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "nnLib.h"

#include "nnAuxillary.h"


struct neuralNetworkLayer * getLastLayer(struct neuralNetwork *network){
   int last = 0;
   struct neuralNetworkLayer *temp;

   if (network->next == NULL){
       last = 1;
       return(NULL);
   }
   else{
       temp = network->next;
   }

   while (last==0){
       if (temp->next == NULL){
           last = 1;
           return(temp);
       }
       else{
           temp = temp->next;
       }
   }
   return(NULL);
}

void createNetwork(struct neuralNetwork *network, int v1){
    int i;

    network->nPerceptrons = v1;
    network->layerID = 0;
    network->perceptrons = malloc(v1*sizeof(double));
    network->next = NULL;
    
    network->poissonLoss = 0; //dont use poisson loss function by default

    for (i = 0; i < v1; i++){
        network->perceptrons[i] = 0.0;
    }

}

int addLayer(struct neuralNetwork *network, int v1, double (*v2)(double), double (*v3)(double)){
    int i;

    struct neuralNetworkLayer *newLayer;
    struct neuralNetworkLayer *temp;


    newLayer = (struct neuralNetworkLayer *)malloc(sizeof *newLayer);

    if (network->next == NULL){
        newLayer->nPerceptrons = v1;
        newLayer->layerID = 1;
        newLayer->lastLayer = 1;
        //newLayer->dropOutActive = 0;

        newLayer->activationFunction = v2;
        newLayer->derrivActivationFunction = v3;

        newLayer->perceptrons = malloc(v1*sizeof(double));
        newLayer->biases = malloc(v1*sizeof(double));
        newLayer->biasesOld = malloc(v1*sizeof(double));
        newLayer->biasesGradient = malloc(v1*sizeof(double));
        newLayer->biasesAggrGradient = malloc(v1*sizeof(double));
        newLayer->bVelocity = malloc(v1*sizeof(double));
        newLayer->bR = malloc(v1*sizeof(double));
        newLayer->bS = malloc(v1*sizeof(double));
        newLayer->inSignal = malloc(v1*sizeof(double));

        //newLayer->dropedOutPerceptronsValue = malloc(v1*sizeof(double));
        //newLayer->dropedOutWeightsValue = malloc(v1*(network->nPerceptrons)*sizeof(double));
        //newLayer->dropedOutPerceptronsID = malloc(v1*sizeof(int));

        newLayer->weightStates = malloc(v1*(network->nPerceptrons)*sizeof(char));
        newLayer->biasStates = malloc(v1*sizeof(char));
        newLayer->isInput = malloc(v1*sizeof(char));
        
        newLayer->weights = malloc(v1*(network->nPerceptrons)*sizeof(double));
        newLayer->weightsOld = malloc(v1*(network->nPerceptrons)*sizeof(double));
        newLayer->weightsGradient = malloc(v1*(network->nPerceptrons)*sizeof(double));
        newLayer->weightsAggrGradient = malloc(v1*(network->nPerceptrons)*sizeof(double));
        newLayer->wVelocity = malloc(v1*(network->nPerceptrons)*sizeof(double));
        newLayer->wR = malloc(v1*(network->nPerceptrons)*sizeof(double));
        newLayer->wS = malloc(v1*(network->nPerceptrons)*sizeof(double));

        newLayer->partialResults = malloc(v1*NPARTIALRESULTS*sizeof(double));

        newLayer->softMax = NULL;

        newLayer->next = NULL;

        for (i = 0; i < v1; i++){
            newLayer->perceptrons[i] = 0.0;
            newLayer->biases[i] = 0.0;
            newLayer->biasesOld[i] = 0.0;  
            newLayer->biasesGradient[i] = 0.0;
            newLayer->biasesAggrGradient[i] = 0.0;
            newLayer->bVelocity[i] = 0.0;
            newLayer->bR[i] = 0.0;
            newLayer->bS[i] = 0.0;
            newLayer->inSignal[i] = 0.0;
            //newLayer->dropedOutPerceptronsValue[i] = 0.0;
            //newLayer->dropedOutPerceptronsID[i] = 0;
            newLayer->biasStates[i] = 0;
            newLayer->isInput[i] = 0;
        }

        for (i = 0; i < v1*(network->nPerceptrons); i++){
            newLayer->weights[i] = 0.0;
            newLayer->weightsOld[i] = 0.0;
            newLayer->weightsGradient[i] = 0.0;
            newLayer->weightsAggrGradient[i] = 0.0;
            newLayer->wVelocity[i] = 0.0;
            newLayer->wR[i] = 0.0;
            newLayer->wS[i] = 0.0;
            //newLayer->dropedOutWeightsValue[i] = 0.0;
            newLayer->weightStates[i] = 0;
 
        }

        for (i = 0; i < v1*NPARTIALRESULTS; i++){
            newLayer->partialResults[i] = 0.0;
        }

        network->next = newLayer;
    }
    else{
        temp = getLastLayer(network);
        if (temp == NULL){
            printf("error, getLastLayer() returns NULL");
            return(1);
        }

        newLayer->nPerceptrons = v1;
        newLayer->layerID = temp->layerID + 1;
        newLayer->lastLayer = 1;
        //newLayer->dropOutActive = 0;

        newLayer->activationFunction = v2;
        newLayer->derrivActivationFunction = v3;

        newLayer->perceptrons = malloc(v1*sizeof(double));
        newLayer->biases = malloc(v1*sizeof(double));
        newLayer->biasesOld = malloc(v1*sizeof(double));
        newLayer->biasesGradient = malloc(v1*sizeof(double));
        newLayer->biasesAggrGradient = malloc(v1*sizeof(double));
        newLayer->bVelocity = malloc(v1*sizeof(double));
        newLayer->bR = malloc(v1*sizeof(double));
        newLayer->bS = malloc(v1*sizeof(double));
        newLayer->inSignal = malloc(v1*sizeof(double));


        newLayer->weights = malloc(v1*(temp->nPerceptrons)*sizeof(double));
        newLayer->weightsOld = malloc(v1*(temp->nPerceptrons)*sizeof(double));
        newLayer->weightsGradient = malloc(v1*(temp->nPerceptrons)*sizeof(double));
        newLayer->weightsAggrGradient = malloc(v1*(temp->nPerceptrons)*sizeof(double));
        newLayer->wVelocity = malloc(v1*(temp->nPerceptrons)*sizeof(double));
        newLayer->wR = malloc(v1*(temp->nPerceptrons)*sizeof(double));
        newLayer->wS = malloc(v1*(temp->nPerceptrons)*sizeof(double));

        //newLayer->dropedOutPerceptronsValue = malloc(v1*sizeof(double));
        //newLayer->dropedOutWeightsValue = malloc(v1*(network->nPerceptrons)*sizeof(double));
        //newLayer->dropedOutPerceptronsID = malloc(v1*sizeof(int));

        newLayer->weightStates = malloc(v1*(temp->nPerceptrons)*sizeof(char));
        newLayer->biasStates = malloc(v1*sizeof(char));
        newLayer->isInput = malloc(v1*sizeof(char));
        


        newLayer->partialResults = malloc(v1*NPARTIALRESULTS*sizeof(double));

        newLayer->softMax = NULL;

        newLayer->next = NULL;
        newLayer->previous = temp;

        for (i = 0; i < v1; i++){
            newLayer->perceptrons[i] = 0.0;
            newLayer->biases[i] = 0.0;
            newLayer->biasesOld[i] = 0.0;
            newLayer->biasesGradient[i] = 0.0;
            newLayer->biasesAggrGradient[i] = 0.0;  
            newLayer->bVelocity[i] = 0.0;
            newLayer->bR[i] = 0.0;
            newLayer->bS[i] = 0.0;
            newLayer->inSignal[i] = 0.0;
            //newLayer->dropedOutPerceptronsValue[i] = 0.0;
            //newLayer->dropedOutPerceptronsID[i] = 0;
            newLayer->biasStates[i] = 0;
            newLayer->isInput[i] = 0;
        }

        for (i = 0; i < v1*(temp->nPerceptrons); i++){
            newLayer->weights[i] = 0.0;
            newLayer->weightsOld[i] = 0.0;
            newLayer->weightsGradient[i] = 0.0;
            newLayer->weightsAggrGradient[i] = 0.0;
            newLayer->wVelocity[i] = 0.0;
            newLayer->wR[i] = 0.0;
            newLayer->wS[i] = 0.0;
            //newLayer->dropedOutWeightsValue[i] = 0.0;
            newLayer->weightStates[i] = 0;
        }

        for (i = 0; i < v1*NPARTIALRESULTS; i++){
            newLayer->partialResults[i] = 0.0;
        }

        temp->lastLayer = 0;
        temp->next = newLayer;
    }    
    return(0);
}

int setSoftMax(struct neuralNetwork *network, int *softMaxIDs){
    struct neuralNetworkLayer *temp;
    int i;

    if (network->next == NULL){
        printf("only input layer exists.\n");
        return(1);
    }
    temp = getLastLayer(network);
    if (temp == NULL){
        printf("error, getLastLayer() returns NULL");
        return(1);
    }

    if (temp->softMax == NULL){
        temp->softMax = malloc((temp->nPerceptrons)*sizeof(int));
    }
    for (i = 0; i < temp->nPerceptrons; i++){
        temp->softMax[i] = softMaxIDs[i];
    }
    return(0);
}

void viewNetwork(struct neuralNetwork *network){
    struct neuralNetworkLayer *temp;
    int i;
    int j;
    int currentLayer = 0;
    int last = 0;

    printf("InputLayer:\n");
    printf("----------------------------------\n");
    printf("#inputs: %d \n", network->nPerceptrons);
    printf("Perceptrons content: \n[");
    for(i = 0; i < network->nPerceptrons; i++){
        printf("%f, ", network->perceptrons[i]);
    }
    printf("]\n\n");

    while(last == 0){
        if (currentLayer == 0){
            if (network->next != NULL){
                temp = network->next;
                currentLayer++;
            }
            else{
                last = 1;
            }
        }
        else{
            if (temp->next != NULL){
                temp = temp->next;
                currentLayer++;
            }
            else{
                last = 1;
            }
        }
        if (last == 0){
            printf("Layer: %d\n", temp->layerID);
            printf("----------------------------------\n");
            printf("#perceptrons: %d\n", temp->nPerceptrons);
            printf("Perceptron contents: \n[");
            for(i = 0; i < temp->nPerceptrons; i++){
                printf("%f, ", temp->perceptrons[i]);
            }
            printf("]\n\nThresholds: \n[");
            for(i = 0; i < temp->nPerceptrons; i++){
                printf("%f, ", temp->biases[i]);
            }
            printf("]\n\nThresholds gradient: \n[");
            for(i = 0; i < temp->nPerceptrons; i++){
                printf("%f, ", temp->biasesGradient[i]);
            }
            printf("]\n\nThresholds states: \n[");
            for(i = 0; i < temp->nPerceptrons; i++){
                printf("%d, ", temp->biasStates[i]);
            }

/*
            printf("]\n\nPartial Results: \n1: [");
            for (i = 0; i < NPARTIALRESULTS; i++){
                for (j = 0; j < temp->nPerceptrons; j++){
                    printf("%f, ", temp->partialResults[i*(temp->nPerceptrons) + j]);
                }
                if (i == NPARTIALRESULTS-1){
                    printf("]\n\nWeights: \n[");
                }
                else{
                    printf("]\n%d: [",i+2);
                }
            }
*/
            printf("]\n\nWeights: \n[");

            if (temp->layerID == 1){
                for(i = 0; i < network->nPerceptrons; i++){
                    for(j = 0; j < temp->nPerceptrons; j++){
                        printf("%f, ", temp->weights[i*(temp->nPerceptrons)+j]);

                    }
                    printf("\n");
                }
            }
            else{
                for(i = 0; i < temp->previous->nPerceptrons; i++){
                    for(j = 0; j < temp->nPerceptrons; j++){
                        printf("%f, ", temp->weights[i*(temp->nPerceptrons)+j]);
                    }
                    printf("\n");
                }
            }


            printf("]\n\nWeights gradient\n[");
            if (temp->layerID == 1){
                for(i = 0; i < network->nPerceptrons; i++){
                    for(j = 0; j < temp->nPerceptrons; j++){
                        printf("%f, ", temp->weightsGradient[i*(temp->nPerceptrons)+j]);

                    }
                    printf("\n");
                }
            }
            else{
                for(i = 0; i < temp->previous->nPerceptrons; i++){
                    for(j = 0; j < temp->nPerceptrons; j++){
                        printf("%f, ", temp->weightsGradient[i*(temp->nPerceptrons)+j]);
                    }
                    printf("\n");
                }
            }
            //printf("]\n\n");

            printf("]\n\nWeights States\n[");
            if (temp->layerID == 1){
                for(i = 0; i < network->nPerceptrons; i++){
                    for(j = 0; j < temp->nPerceptrons; j++){
                        printf("%d, ", (int)temp->weightStates[i*(temp->nPerceptrons)+j]);

                    }
                    printf("\n");
                }
            }
            else{
                for(i = 0; i < temp->previous->nPerceptrons; i++){
                    for(j = 0; j < temp->nPerceptrons; j++){
                        printf("%d, ", temp->weightStates[i*(temp->nPerceptrons)+j]);
                    }
                    printf("\n");
                }
            }
            printf("]\n\n");

        }
        if (last == 1){
            printf("Softmax: ");
            if (temp->softMax == NULL){
                printf("No\n\n");
            }
            else{
                printf("Yes\n[%d", temp->softMax[0]);
                for(i = 1 ; i < temp->nPerceptrons; i++){
                    printf(", %d", temp->softMax[i]);
                }
                printf("]\n\n");
            }
        }
    }
}

int viewOutput(struct neuralNetwork *network){
    struct neuralNetworkLayer *temp;
    int i;

    if (network->next == NULL){
        printf("only input layer exists.\n");
        return(1);
    }
    temp = getLastLayer(network);
    if (temp == NULL){
        printf("error, getLastLayer() returns NULL");
        return(1);
    }
    printf("Outputs:\n[");
    for (i = 0; i < temp->nPerceptrons; i++){
        printf("%f, ", temp->perceptrons[i]);
    }
    printf("]\n");
    return(0);
}

int getOutput(struct neuralNetwork *network, double *output){
    struct neuralNetworkLayer *temp;
    int i;

    if (network->next == NULL){
        printf("only input layer exists.\n");
        return(1);
    }
    temp = getLastLayer(network);
    if (temp == NULL){
        printf("error, getLastLayer() returns NULL");
        return(1);
    }

    for (i = 0; i < temp->nPerceptrons; i++){
        output[i] = temp->perceptrons[i];
    }

    return(0);
}


void setWeights(struct neuralNetwork *network, double (*setToThis)(void)){
    int i;
    int j;
    int last = 0;
    struct neuralNetworkLayer *temp;

    if (network->next == NULL){
        last = 1;
    }
    else{
        temp = network->next;
    }

    while(last == 0){
        if (temp->layerID == 1){
            for(i = 0; i < network->nPerceptrons; i++){
                for(j = 0; j < temp->nPerceptrons; j++){
                    temp->weights[i*(temp->nPerceptrons)+j] = setToThis();
                }
            }

        }
        else{
            for(i = 0; i < temp->previous->nPerceptrons; i++){
                for(j = 0; j < temp->nPerceptrons; j++){
                    temp->weights[i*(temp->nPerceptrons)+j] = setToThis();
                }
            }
        }

        if (temp->next == NULL){
            last = 1;
        }
        else{
            temp = temp->next;
        }
    }
}

void setWeightsRandom(struct neuralNetwork *network){
    int i;
    int j;
    int last = 0;
    struct neuralNetworkLayer *temp;

    

    if (network->next == NULL){
        last = 1;
    }
    else{
        temp = network->next;
    }

    while(last == 0){
        if (temp->layerID == 1){
            for(i = 0; i < network->nPerceptrons; i++){
                for(j = 0; j < temp->nPerceptrons; j++){
                    temp->weights[i*(temp->nPerceptrons)+j] = rand()/(RAND_MAX*1.0);
                }
            }

        }
        else{
            for(i = 0; i < temp->previous->nPerceptrons; i++){
                for(j = 0; j < temp->nPerceptrons; j++){
                    temp->weights[i*(temp->nPerceptrons)+j] = rand()/(RAND_MAX*1.0);
                }
            }
        }

        if (temp->next == NULL){
            last = 1;
        }
        else{
            temp = temp->next;
        }
    }
}


void setBiases(struct neuralNetwork *network, double (*setToThis)(void)){
    int i;
    int last = 0;
    struct neuralNetworkLayer *temp;



    if (network->next == NULL){
        last = 1;
    }
    else{
        temp = network->next;
    }

    while(last == 0){
        for(i = 0; i < temp->nPerceptrons; i++){
            temp->biases[i] = setToThis();
        }

        if (temp->next == NULL){
            last = 1;
        }
        else{
            temp = temp->next;
        }
    }
}

void setThresholdsRandom(struct neuralNetwork *network){
    int i;
    int last = 0;
    struct neuralNetworkLayer *temp;

    

    if (network->next == NULL){
        last = 1;
    }
    else{
        temp = network->next;
    }

    while(last == 0){
        for(i = 0; i < temp->nPerceptrons; i++){
            temp->biases[i] = (rand()/(RAND_MAX*1.0));
        }

        if (temp->next == NULL){
            last = 1;
        }
        else{
            temp = temp->next;
        }
    }
}

int setWeightStates(struct neuralNetwork *network, int layer, int connectionThis, int connectionPrevious, int setTo){
    int i;
    int last = 0;
    char cSetTo;

    struct neuralNetworkLayer *temp;

    cSetTo = (char)setTo;

    if (network->next == NULL){
        printf("Error in setWeightStates(): only input layer exists\n");
        return(1);
    }
    else{
        temp = network->next;
        if (temp->next == NULL){
            printf("Error in setWeightStates(): there are no hidden layers\n");
            return(2);
        }
    }

    while(last == 0){
        if (temp->layerID == layer){
            last = 1;
        }
        else{
            if(temp->next == NULL){
                printf("Error in setWeightStates(): could not reach layer %d, there are only %d layers", layer, temp->layerID);
                return(3);
            }
            else{
                temp = temp->next;
            }
        }
    }

    //for (i = 0; i < temp->nPerceptrons; i++){
    //    temp->dropedOutPerceptronsID[i] = dropOutIDs[i];
    //}

    temp->weightStates[connectionPrevious*(temp->nPerceptrons) + connectionThis] = cSetTo;

    return(0);
}


int setBiasStates(struct neuralNetwork *network, int layer, int neuron, int setTo){
    int i;
    int last = 0;
    char cSetTo;

    struct neuralNetworkLayer *temp;

    cSetTo = (char)setTo;

    if (network->next == NULL){
        printf("Error in setBiasStates(): only input layer exists\n");
        return(1);
    }
    else{
        temp = network->next;
        if (temp->next == NULL){
            printf("Error in setBiasStates(): there are no hidden layers\n");
            return(2);
        }
    }

    while(last == 0){
        if (temp->layerID == layer){
            last = 1;
        }
        else{
            if(temp->next == NULL){
                printf("Error in setBiasStates(): could not reach layer %d, there are only %d layers", layer, temp->layerID);
                return(3);
            }
            else{
                temp = temp->next;
            }
        }
    }

    //for (i = 0; i < temp->nPerceptrons; i++){
    //    temp->dropedOutPerceptronsID[i] = dropOutIDs[i];
    //}

    temp->biasStates[neuron] = cSetTo;

    return(0);
}

int setInput(struct neuralNetwork *network, int layer, int neuron){
    int i;
    int j;
    int last = 0;
    int loopLength;

    struct neuralNetworkLayer *temp;


    if (network->next == NULL){
        printf("Error in setInput(): only input layer exists\n");
        return(1);
    }
    else{
        temp = network->next;
        if (temp->next == NULL){
            printf("Error in setInput(): there are no hidden layers\n");
            return(2);
        }
    }
 
    //go to the right layer
    while(last == 0){
        if (temp->layerID == layer){
            last = 1;
        }
        else{
            if(temp->next == NULL){
                printf("Error in setInput(): could not reach layer %d, there are only %d layers", layer, temp->layerID);
                return(3);
            }
            else{
                temp = temp->next;
            }
        }
    }


    //Set mark that this is an input neuron
    temp->isInput[neuron] = 1;


    //Turn off all weights from the previous layer to this input neuron
    if (temp->layerID == 1){
        loopLength = network->nPerceptrons;
    }
    else{
        loopLength = temp->previous->nPerceptrons;
    }

    for(i = 0; i < loopLength; i++){
        temp->weightStates[i*(temp->nPerceptrons)+neuron] = 1;
    }

    //Turn off the bias term to this input
    temp->biasStates[neuron] = 1;

    return(0);
}

int setPoissonLoss(struct neuralNetwork *network){
    network->poissonLoss = 1;
    return(0);
}

void aggregateGradient(struct neuralNetwork *network, int action, double operand){
    int i;
    int j;
    int last = 0;
    struct neuralNetworkLayer *temp;


    //Actions: 
    //1-aggregate                               AGGRGRAD_AGGREGATE
    //2-set to zero                             AGGRGRAD_ZERO
    //3-pull                                    AGGRGRAD_PULL
    //4-divide aggregated by operand            AGGRGRAD_DIVIDE
    //5-divide aggregated by operand and pull   AGGRGRAD_DIVPULL



    if (network->next == NULL){
        printf("Error in aggregateGradient(): only input layer exists\n");
        last = 1;
    }
    else{
        temp = network->next;
    }

    while(last == 0){
        if (temp->layerID == 1){
            for(i = 0; i < network->nPerceptrons; i++){
                for(j = 0; j < temp->nPerceptrons; j++){
                    if (action == 1){
                        temp->weightsAggrGradient[i*(temp->nPerceptrons)+j] = temp->weightsAggrGradient[i*(temp->nPerceptrons)+j] + temp->weightsGradient[i*(temp->nPerceptrons)+j];
                    }
                    else if (action == 2){
                        temp->weightsAggrGradient[i*(temp->nPerceptrons)+j] = 0.0;
                    }
                    else if (action == 3){
                        temp->weightsGradient[i*(temp->nPerceptrons)+j] = temp->weightsAggrGradient[i*(temp->nPerceptrons)+j];
                    }
                    else if (action == 4){
                        temp->weightsAggrGradient[i*(temp->nPerceptrons)+j] /= operand;
                    }
                    else if (action == 5){
                        temp->weightsAggrGradient[i*(temp->nPerceptrons)+j] = (temp->weightsAggrGradient[i*(temp->nPerceptrons)+j])/(1.0*operand);
                        temp->weightsGradient[i*(temp->nPerceptrons)+j] = temp->weightsAggrGradient[i*(temp->nPerceptrons)+j];
                    }

                }
            }
            for(j = 0; j < temp->nPerceptrons; j++){
                if (action == 1){
                    temp->biasesAggrGradient[j] = temp->biasesAggrGradient[j] + temp->biasesGradient[j];
                }
                else if (action == 2){
                    temp->biasesAggrGradient[j] = 0.0;
                }
                else if (action == 3){
                    temp->biasesGradient[j] = temp->biasesAggrGradient[j];
                }
                else if (action == 4){
                    temp->biasesAggrGradient[j] /= operand;
                }
                else if (action == 5){
                    temp->biasesAggrGradient[j] = (temp->biasesAggrGradient[j])/(1.0*operand);
                    temp->biasesGradient[j] = temp->biasesAggrGradient[j];
                }

            }
        }
        else{
            for(i = 0; i < temp->previous->nPerceptrons; i++){
                for(j = 0; j < temp->nPerceptrons; j++){
                    if (action == 1){
                        temp->weightsAggrGradient[i*(temp->nPerceptrons)+j] = temp->weightsAggrGradient[i*(temp->nPerceptrons)+j] + temp->weightsGradient[i*(temp->nPerceptrons)+j];
                    }
                    else if (action == 2){
                        temp->weightsAggrGradient[i*(temp->nPerceptrons)+j] = 0.0;
                    }
                    else if (action == 3){
                        temp->weightsGradient[i*(temp->nPerceptrons)+j] = temp->weightsAggrGradient[i*(temp->nPerceptrons)+j];
                    }
                    else if (action == 4){
                        temp->weightsAggrGradient[i*(temp->nPerceptrons)+j] /= operand;
                    }
                    else if (action == 5){
                        temp->weightsAggrGradient[i*(temp->nPerceptrons)+j] = (temp->weightsAggrGradient[i*(temp->nPerceptrons)+j])/(1.0*operand);
                        temp->weightsGradient[i*(temp->nPerceptrons)+j] = temp->weightsAggrGradient[i*(temp->nPerceptrons)+j];
                    }

                }
            }
            for(j = 0; j < temp->nPerceptrons; j++){
                if (action == 1){
                    temp->biasesAggrGradient[j] = temp->biasesAggrGradient[j] + temp->biasesGradient[j];
                }
                else if (action == 2){
                    temp->biasesAggrGradient[j] = 0.0;
                }
                else if (action == 3){
                    temp->biasesGradient[j] = temp->biasesAggrGradient[j];
                }
                else if (action == 4){
                    temp->biasesAggrGradient[j] /= operand;
                }
                else if (action == 5){
                    temp->biasesAggrGradient[j] = (temp->biasesAggrGradient[j])/(1.0*operand);
                    temp->biasesGradient[j] = temp->biasesAggrGradient[j];
                }

            }
        }

        if (temp->next == NULL){
            last = 1;
        }
        else{
            temp = temp->next;
        }
    }
}

int forwardPropagate(struct neuralNetwork *network, double pattern[], double exposure){  //exposure is only used when poisson deviance function is used
    int i; //previous layer (l-1) loop index
    int j; //current layer (l) loop index
    int sm; //different softmaxes loop index

    int idInput = 0; //variable for identifying inputs other than the first layer

    int softMaxOn = 0;
    int nSoftMax;
    int last = 0;
    
    double b = 0.0 ;
 
    struct neuralNetworkLayer *temp;

    if (network->next == NULL){
        last = 1;
        printf("Error in forwardPropagate(): only input layer exists\n");
        return(1);
    }
    else{
        temp = network->next;
    }

    for (j = 0; j < network->nPerceptrons; j++){
        network->perceptrons[j] = pattern[j];
    }


    temp = getLastLayer(network);    //check if softmax is active, if yes: then check number of softmaxes
    if (temp->softMax == NULL){
        softMaxOn = 0;
    }
    else{
        softMaxOn = 1;
        nSoftMax = arrMax_int(temp->softMax, temp->nPerceptrons);
    }
    temp = network->next;

    while(last == 0){
        if (temp->layerID == 1){
            for(j = 0; j < temp->nPerceptrons; j++){
                if (temp->isInput[j] == 1){                  //if this is an interior input neuron, just set it to the pattern input value
                    temp->perceptrons[j] = pattern[(network->nPerceptrons) + idInput];
                    temp->inSignal[j] = 0;
                    idInput += 1;
                }
                else{
                    for(i = 0; i < network->nPerceptrons; i++){
                        if (temp->weightStates[i*(temp->nPerceptrons)+j] == 0){
                            b = b + (temp->weights[i*(temp->nPerceptrons)+j])*(network->perceptrons[i]);
                        }
                    }
                    if (temp->biasStates[j] == 0){ 
                        b = b - temp->biases[j];
                    }
                    temp->inSignal[j] = b;
                    temp->perceptrons[j] = temp->activationFunction(b);
                    b = 0.0;
                }
            }
            if ((softMaxOn == 1)&&(temp->next == NULL)){  //if softmax is on, and this is the last layer
                for (sm = 1; sm < (nSoftMax+1); sm++){
                    b = 0.0;
                    for (j = 0; j < temp->nPerceptrons; j++){
                        if (temp->softMax[j] == sm){
                            b += exp(temp->inSignal[j]);
                        }
                    }
                for (j = 0; j < temp->nPerceptrons; j++){
                        if (temp->softMax[j] == sm){
                            temp->perceptrons[j] = exp(temp->inSignal[j])/b;
                        }
                    }
                }
            }
            else if((network->poissonLoss == 1)&&(temp->next == NULL)){   //if this is the last layer and use poisson deviance function
                temp->perceptrons[0] = exp(exposure + temp->inSignal[0]);
                if (temp->nPerceptrons > 1){
                    printf("warning in forwardPropagate(): there are more than 1 output, but using Poisson deviance loss function.\n");
                }
            }
        }  //end layerID if statement
        else{
            for(j = 0; j < temp->nPerceptrons; j++){
                if (temp->isInput[j] == 1){                  //if this is an interior input neuron, just set it to the pattern input value
                    temp->perceptrons[j] = pattern[(network->nPerceptrons) + idInput];
                    temp->inSignal[j] = 0;
                    idInput += 1;
                }
                else{
                    for(i = 0; i < temp->previous->nPerceptrons; i++){
                        if (temp->weightStates[i*(temp->nPerceptrons)+j] == 0){
                            b = b + (temp->weights[i*(temp->nPerceptrons)+j])*(temp->previous->perceptrons[i]);
                        }
                    }
                    if (temp->biasStates[j] == 0){ 
                        b = b - temp->biases[j];
                    }
                    temp->inSignal[j] = b;
                    temp->perceptrons[j] = temp->activationFunction(b);
                    b = 0.0;
                }
            }
            if ((softMaxOn == 1)&&(temp->next == NULL)){               //if softmax is on, and this is the last layer
                for (sm = 1; sm < (nSoftMax+1); sm++){
                    b = 0.0;
                    for (j = 0; j < temp->nPerceptrons; j++){
                        if (temp->softMax[j] == sm){
                            b += exp(temp->inSignal[j]);
                        }
                    }
                    for (j = 0; j < temp->nPerceptrons; j++){
                        if (temp->softMax[j] == sm){
                            temp->perceptrons[j] = exp(temp->inSignal[j])/b;
                        }
                    }
                }
            }
            else if((network->poissonLoss == 1)&&(temp->next == NULL)){        //if this is the last layer and use poisson deviance function
                temp->perceptrons[0] = exp(exposure + temp->inSignal[0]);
                if (temp->nPerceptrons > 1){
                    printf("warning in forwardPropagate(): there are more than 1 output, but using Poisson deviance loss function.\n");
                }
            }
        }

        if (temp->next == NULL){
            last = 1;
        }
        else{
            temp = temp->next;
        }
    }
    return(0);
}

int gradient(struct neuralNetwork *network, double targetPattern[]){

   struct neuralNetworkLayer *temp;
   int i; //previous layer (l-1) loop index
   int j; //current layer (l) loop index
   int k; //next layer (l+1) loop index
   int jj; //current layer (l) loop index (used for softmax)
   int sm; //different softmaxes loop index

   int softMaxOn = 0;
   //int nSoftMax;

   int last = 0;
   double b;
   double d;

   
   if (network->next == NULL){
       printf("Error, only input layer exists.\n");
       return(1);
   }

   temp = getLastLayer(network);

   if (temp == NULL){
       printf("Error, getLastLayer() returns NULL.\n");
       return(1);
   }

   if (temp->softMax == NULL){
       softMaxOn = 0;
   }
   else{
       softMaxOn = 1;
       //nSoftMax = arrMax_int(temp->softMax, temp->nPerceptrons);
   }

   if (temp->layerID == 1){                        //case with no hidden layers
       for (j = 0; j < temp->nPerceptrons; j++){
           //for(i = 0; i < network->nPerceptrons; i++){
           //    b = b + (temp->weights[i*(temp->nPerceptrons)+j])*(network->perceptrons[i]);
           //}
           //b = b - temp->biases[j];
           
           
           //This is the output layer so there are different cases depending on what loss function is used
           //Three cases: softmax, poisson deviance loss, squared distance loss function (normal case)

           if (softMaxOn == 1){          
               b = 0.0;
               for (jj = 0; jj < temp->nPerceptrons; jj++){
                   if (temp->softMax[j] == temp->softMax[jj]){
                       b += targetPattern[jj];
                   }
               }
               temp->partialResults[j] = targetPattern[j] - b*(temp->perceptrons[j]);
           }
           else if (network->poissonLoss == 1){

//               temp->partialResults[j] = 2*(temp->perceptrons[j] - targetPattern[j]);
               temp->partialResults[j] = 2*(targetPattern[j] - temp->perceptrons[j]);                 //trodde det skulle vara tvärtom när jag kolla beräkningarna, men detta blir rätt när man kör programmet.

               if (temp->nPerceptrons > 1){
                   printf("Warning in gradient(): more than 1 neuron in output layer, but using Poisson deviance loss\n");
               }
           }
           else{                                            
               b = temp->inSignal[j];
               temp->partialResults[j] = temp->derrivActivationFunction(b)*(targetPattern[j] - temp->perceptrons[j]);
           }


           d = temp->partialResults[j];
           for (i = 0; i < network->nPerceptrons; i++){
               if (temp->weightStates[i*(temp->nPerceptrons)+j] == 0){
                   temp->weightsGradient[i*(temp->nPerceptrons) + j] = -d*(network->perceptrons[i]);
               }
           }
           if (temp->biasStates[j] == 0){ 
               temp->biasesGradient[j] = d;
           }
           d = 0.0;
           b = 0.0;
       }
   }
   else{
       for (j = 0; j < temp->nPerceptrons; j++){
           //for(i = 0; i < temp->previous->nPerceptrons; i++){
           //    b = b + (temp->weights[i*(temp->nPerceptrons)+j])*(temp->previous->perceptrons[i]);
           //}
           //b = b - temp->biases[j];
           //temp->partialResults[j] = temp->derrivActivationFunction(b)*(targetPattern[j] - temp->perceptrons[j]);

 
           //This is the output layer so there are different cases depending on what loss function is used
           //Three cases: softmax, poisson deviance loss, squared distance loss function (normal case)

           if (softMaxOn == 1){          
               b = 0.0;
               for (jj = 0; jj < temp->nPerceptrons; jj++){
                   if (temp->softMax[j] == temp->softMax[jj]){
                       b += targetPattern[jj];
                   }
               }
               temp->partialResults[j] = targetPattern[j] - b*(temp->perceptrons[j]);
           }
           else if (network->poissonLoss == 1){
//               temp->partialResults[j] = 2*(temp->perceptrons[j] - targetPattern[j]);
               temp->partialResults[j] = 2*(targetPattern[j] - temp->perceptrons[j]);              //trodde det skulle vara tvärtom när jag kolla beräkningarna, men detta blir rätt när man kör programmet.

               if (temp->nPerceptrons > 1){
                   printf("Warning in gradient(): more than 1 neuron in output layer, but using Poisson deviance loss\n");
               }
           }
           else{                                            
               b = temp->inSignal[j];
               temp->partialResults[j] = temp->derrivActivationFunction(b)*(targetPattern[j] - temp->perceptrons[j]);
           }

           d = temp->partialResults[j];
           for (i = 0; i < temp->previous->nPerceptrons; i++){
               if (temp->weightStates[i*(temp->nPerceptrons)+j] == 0){
                   temp->weightsGradient[i*(temp->nPerceptrons) + j] = -d*(temp->previous->perceptrons[i]);
               }
           }
           if (temp->biasStates[j] == 0){ 
               temp->biasesGradient[j] = d;
           }
           d = 0.0;
           b = 0.0;
       }
       last = 0;

       while(last == 0){
           temp = temp->previous;
           if (temp->layerID == 1){

               for (j = 0; j < temp->nPerceptrons; j++){
                   //b = 0.0;
                   //for(i = 0; i < network->nPerceptrons; i++){
                   //    b = b + (temp->weights[i*(temp->nPerceptrons)+j])*(network->perceptrons[i]);
                   //}
                   //b = b - temp->biases[j];

                   b = temp->inSignal[j];

                   d = 0.0;
                   for (k = 0; k < temp->next->nPerceptrons; k++){
                       if (temp->next->weightStates[j*(temp->next->nPerceptrons) + k] == 0){
                           d = d + (temp->next->partialResults[k])*(temp->next->weights[j*(temp->next->nPerceptrons) + k])*(temp->derrivActivationFunction(b));
                       }
                   }
                   temp->partialResults[j] = d;

                   for (i = 0; i < network->nPerceptrons; i++){
                       if (temp->weightStates[i*(temp->nPerceptrons) + j] == 0){
                           temp->weightsGradient[i*(temp->nPerceptrons) + j] = -d*(network->perceptrons[i]);
                       }
                   }
                   if (temp->biasStates[j] == 0){ 
                       temp->biasesGradient[j] = d;
                   }
                   b = 0.0;
               }
               last = 1;
           }
           else{
               for (j = 0; j < temp->nPerceptrons; j++){
                   //for(i = 0; i < temp->previous->nPerceptrons; i++){
                   //    b = b + (temp->weights[i*(temp->nPerceptrons)+j])*(temp->previous->perceptrons[i]);
                   //}
                   //b = b - temp->biases[j];

                   b = temp->inSignal[j];

                   d = 0.0;
                   for (k = 0; k < temp->next->nPerceptrons; k++){
                       if (temp->next->weightStates[j*(temp->next->nPerceptrons) + k] == 0){
                           d = d + (temp->next->partialResults[k])*(temp->next->weights[j*(temp->next->nPerceptrons) + k])*(temp->derrivActivationFunction(b));
                       }
                   }
                   temp->partialResults[j] = d;

                   for (i = 0; i < temp->previous->nPerceptrons; i++){
                       if (temp->weightStates[i*(temp->nPerceptrons) + j] == 0){
                           temp->weightsGradient[i*(temp->nPerceptrons) + j] = -d*(temp->previous->perceptrons[i]);
                       }
                   }
                   if (temp->biasStates[j] == 0){ 
                       temp->biasesGradient[j] = d;
                   }
                   b = 0.0;
               }
           }
       }
   }
   return(0);
}

//Optimization algorithms---------------------------------------------------------------------------------------------------------

//not implemented setting weights states and bias states
int stochastic_gradient_descent(struct neuralNetwork *network, double patterns[], double targets[], int nPatterns, int patternSize, int targetSize, int nEpochs, double eta){
    int i;
    int j;
    int k;
    int ep;
    int last = 0;
    int currentIndex;
    double *currentPattern;
    double *currentTarget;
    struct neuralNetworkLayer *temp;

    currentPattern = malloc(patternSize*sizeof(double));
    currentTarget = malloc(targetSize*sizeof(double));
   
    for (ep = 0; ep < nEpochs; ep++){
        currentIndex = rand()%nPatterns;
        getArray(patterns, currentPattern, currentIndex, patternSize);
        getArray(targets, currentTarget, currentIndex, targetSize);

        forwardPropagate(network, currentPattern,0.0);
        gradient(network, currentTarget);

        temp = network->next;
        if (temp == NULL){
            printf("Network has only input layer\n");
            free(currentPattern);
            free(currentTarget);
            return(1);
        }

        last = 0;
        while(last == 0){                       //update weights and threshold (called biases) terms
            if (temp->layerID == 1){
           
                for (j = 0; j < temp->nPerceptrons; j++){
                    for (i = 0; i < network->nPerceptrons; i++){
                        temp->weights[i*(temp->nPerceptrons) + j] -= eta*(temp->weightsGradient[i*(temp->nPerceptrons) + j]);
                    }
                    temp->biases[j] -= eta*(temp->biasesGradient[j]);
                }    
            }
            else{
                for (j = 0; j < temp->nPerceptrons; j++){
                    for (i = 0; i < temp->previous->nPerceptrons; i++){
                        temp->weights[i*(temp->nPerceptrons) + j] -= eta*(temp->weightsGradient[i*(temp->nPerceptrons) + j]);
                    }
                    temp->biases[j] -= eta*(temp->biasesGradient[j]);
                }
            }
            if (temp->next == NULL){
                last = 1;
            }
            else{
                temp = temp->next;
            }
        } //end while 
    } //end for
    
    free(currentPattern);
    free(currentTarget);
    return(0);
}


//not implemented setting weights states and bias states
int sgd_nestrov(struct neuralNetwork *network, double patterns[], double targets[], int nPatterns, int patternSize, int targetSize, int nEpochs, double eta, double alpha){
    int i;
    int j;
    int ep;
    int last = 0;
    int currentIndex;
    int nPerceptronsPrevious;

    double *currentPattern;
    double *currentTarget;
    struct neuralNetworkLayer *temp;

    currentPattern = malloc(patternSize*sizeof(double));
    currentTarget = malloc(targetSize*sizeof(double));

    temp = network->next;
    if (temp == NULL){
        printf("Network has only input layer\n");
        free(currentPattern);
        free(currentTarget);
        return(1);
    }

    last = 0;
    while(last == 0){                //set velocity to zero before starting to iterate
        if (temp->layerID == 1){
            nPerceptronsPrevious = network->nPerceptrons;
        }
        else{
            nPerceptronsPrevious = temp->previous->nPerceptrons;
        }

        for(j = 0; j < temp->nPerceptrons; j++){
            for (i = 0; i < nPerceptronsPrevious; i++){
                temp->wVelocity[i*(temp->nPerceptrons) + j] = 0.0;
            }
            temp->bVelocity[j] = 0.0;
        }
        if (temp->next == NULL){
            last = 1;
        }
        else{
            temp = temp->next;
        }
    }
    last = 0;   


    for (ep = 0; ep < nEpochs; ep++){        //start epochs loop
        currentIndex = rand()%nPatterns;
        getArray(patterns, currentPattern, currentIndex, patternSize);
        getArray(targets, currentTarget, currentIndex, targetSize);

        forwardPropagate(network, currentPattern,0.0);


        //Apply interim update, and store old weights
        temp = network->next;
        last = 0;
        while(last == 0){               
            if (temp->layerID == 1){
                nPerceptronsPrevious = network->nPerceptrons;
            }
            else{
                nPerceptronsPrevious = temp->previous->nPerceptrons;
            }

            for(j = 0; j < temp->nPerceptrons; j++){
                for (i = 0; i < nPerceptronsPrevious; i++){
                    temp->weightsOld[i*(temp->nPerceptrons) + j] = temp->weights[i*(temp->nPerceptrons) + j];
                    temp->weights[i*(temp->nPerceptrons) + j] += alpha*(temp->wVelocity[i*(temp->nPerceptrons) + j]);
                }
                temp->biasesOld[j] = temp->biases[j];
                temp->biases[j] += alpha*temp->bVelocity[j];
            }
            if (temp->next == NULL){
                last = 1;
            }
            else{
                temp = temp->next;
            }
        }
        last = 0; 

        //compute gradient in interim point
        gradient(network, currentTarget);


        //compute velocity updates and then apply updates to weights and biases
        temp = network->next;
        last = 0;
        while(last == 0){                //set velocity to zero before starting to iterate
            if (temp->layerID == 1){
                nPerceptronsPrevious = network->nPerceptrons;
            }
            else{
                nPerceptronsPrevious = temp->previous->nPerceptrons;
            }

            for(j = 0; j < temp->nPerceptrons; j++){
                for (i = 0; i < nPerceptronsPrevious; i++){
                    temp->wVelocity[i*(temp->nPerceptrons) + j] = alpha*temp->wVelocity[i*(temp->nPerceptrons) + j] - eta*(temp->weightsGradient[i*(temp->nPerceptrons) + j]);
                    temp->weights[i*(temp->nPerceptrons) + j] = temp->weightsOld[i*(temp->nPerceptrons) + j] + temp->wVelocity[i*(temp->nPerceptrons) + j];
                }
                temp->bVelocity[j] = alpha*(temp->bVelocity[j]) - eta*(temp->biasesGradient[j]);
                temp->biases[j] = temp->biasesOld[j] + temp->bVelocity[j];
            }
            if (temp->next == NULL){
                last = 1;
            }
            else{
                temp = temp->next;
            }
        }
    } //end for
    
    free(currentPattern);
    free(currentTarget);
    return(0);
}



//not implemented setting weights states and bias states
int sgd_adam(struct neuralNetwork *network, double patterns[], double targets[], int nPatterns, int patternSize, int targetSize, int nEpochs, double eta, double rho1, double rho2, double delta){
    int i;
    int j;
    int ep;
    int t;
    int last = 0;
    int currentIndex;
    int nPerceptronsPrevious;

    double hatS;
    double hatR;
    double dTheta;

    double *currentPattern;
    double *currentTarget;
    struct neuralNetworkLayer *temp;

    currentPattern = malloc(patternSize*sizeof(double));
    currentTarget = malloc(targetSize*sizeof(double));

    temp = network->next;
    if (temp == NULL){
        printf("Network has only input layer\n");
        free(currentPattern);
        free(currentTarget);
        return(1);
    }

    last = 0;
    t = 0;                           //set time step to zero
    while(last == 0){                //set 1st and 2nd moments variables to zero before starting iteration
        if (temp->layerID == 1){
            nPerceptronsPrevious = network->nPerceptrons;
        }
        else{
            nPerceptronsPrevious = temp->previous->nPerceptrons;
        }

        for(j = 0; j < temp->nPerceptrons; j++){
            for (i = 0; i < nPerceptronsPrevious; i++){
                temp->wR[i*(temp->nPerceptrons) + j] = 0.0;
                temp->wS[i*(temp->nPerceptrons) + j] = 0.0;

            }
            temp->bR[j] = 0.0;
            temp->bS[j] = 0.0;

        }
        if (temp->next == NULL){
            last = 1;
        }
        else{
            temp = temp->next;
        }
    }
    last = 0;   


    for (ep = 0; ep < nEpochs; ep++){        //start iteration
        currentIndex = rand()%nPatterns;
        getArray(patterns, currentPattern, currentIndex, patternSize);
        getArray(targets, currentTarget, currentIndex, targetSize);

        t += 1;
        forwardPropagate(network, currentPattern,0.0);
        gradient(network, currentTarget);
     
        //compute moments updates and then apply updates to weights and biases
        temp = network->next;
        last = 0;
        while(last == 0){                //set velocity to zero before starting to iterate
            if (temp->layerID == 1){
                nPerceptronsPrevious = network->nPerceptrons;
            }
            else{
                nPerceptronsPrevious = temp->previous->nPerceptrons;
            }

            for(j = 0; j < temp->nPerceptrons; j++){
                for (i = 0; i < nPerceptronsPrevious; i++){

                    //update biased 1st moment estimate
                    temp->wS[i*(temp->nPerceptrons) + j] = rho1*(temp->wS[i*(temp->nPerceptrons) + j]) + (1-rho1)*(temp->weightsGradient[i*(temp->nPerceptrons) + j]);

                    //update biased 2nd moment estimate
                    temp->wR[i*(temp->nPerceptrons) + j] = rho2*(temp->wR[i*(temp->nPerceptrons) + j]) + (1-rho2)*(temp->weightsGradient[i*(temp->nPerceptrons) + j])*(temp->weightsGradient[i*(temp->nPerceptrons) + j]);

                    //correct bias in 1st moment estimate
                    hatS = (temp->wS[i*(temp->nPerceptrons) + j])/(1-pow(rho1,t));

                    //correct bias in 2nd moment estimate
                    hatR = (temp->wR[i*(temp->nPerceptrons) + j])/(1-pow(rho2,t));

                    //compute update
                    dTheta = -eta*hatS/(sqrt(hatR)+delta);
                    
                    //apply update
                    temp->weights[i*(temp->nPerceptrons) + j] += dTheta;
                }
                //update biased 1st moment estimate
                temp->bS[j] = rho1*(temp->bS[j]) + (1-rho1)*(temp->biasesGradient[j]);

                //update biased 2nd moment estimate
                temp->bR[j] = rho2*(temp->bR[j]) + (1-rho2)*(temp->biasesGradient[j])*(temp->biasesGradient[j]);

                //correct bias in 1st moment estimate
                hatS = (temp->bS[j])/(1-pow(rho1,t));

                //correct bias in 2nd moment estimate
                hatR = (temp->bR[j])/(1-pow(rho2,t));

                //compute update
                dTheta = -eta*hatS/(sqrt(hatR)+delta);

                //apply update
                temp->biases[j] += dTheta;
            }
            if (temp->next == NULL){
                last = 1;
            }
            else{
                temp = temp->next;
            }
        }
    }
    free(currentPattern);
    free(currentTarget);
    return(0);
}


int sgd_nadam(struct neuralNetwork *network, double patterns[], double targets[], int nPatterns, int patternSize, int targetSize, int batchSize, int nEpochs, double eta, double rho1, double rho2, double epsilon){
    int i;
    int j;
    int ep;
    int t;
    int last = 0;
    int it; //loop variable for iteration for each the epoch
    int currentIndex;
    int nPerceptronsPrevious;

    int p; //loop variable for different patterns in the batch

    double hatS;
    double hatR;
    double dTheta;

    double *currentPattern;
    double *currentTarget;
    int *currentBatchesIndexes;

    struct neuralNetworkLayer *temp;

    currentPattern = malloc(patternSize*sizeof(double));
    currentTarget = malloc(targetSize*sizeof(double));
    currentBatchesIndexes = malloc(nPatterns*sizeof(int));

    temp = network->next;
    if (temp == NULL){
        printf("Network has only input layer\n");
        free(currentPattern);
        free(currentTarget);
        return(1);
    }

    last = 0;
    t = 0;                           //set time step to zero
    while(last == 0){                //set 1st and 2nd moments variables to zero before starting iteration
        if (temp->layerID == 1){
            nPerceptronsPrevious = network->nPerceptrons;
        }
        else{
            nPerceptronsPrevious = temp->previous->nPerceptrons;
        }

        for(j = 0; j < temp->nPerceptrons; j++){
            for (i = 0; i < nPerceptronsPrevious; i++){
                temp->wR[i*(temp->nPerceptrons) + j] = 0.0;
                temp->wS[i*(temp->nPerceptrons) + j] = 0.0;

            }
            temp->bR[j] = 0.0;
            temp->bS[j] = 0.0;

        }
        if (temp->next == NULL){
            last = 1;
        }
        else{
            temp = temp->next;
        }
    }
    last = 0;   

    printf("Epoch (of %d): ", nEpochs); 
    for (ep = 0; ep < nEpochs; ep++){        //start iteration

        //if ((ep+1)%100 == 0){
            printf("%d, ",(ep+1));
        //}
    
        //Pick index order that will make up the batches of this epoch

        generateOrder(currentBatchesIndexes, nPatterns);

        if (nPatterns%batchSize != 0){
            printf("Warning from sgd_nadam(): (#patterns)/(batch size) is not an integer. Some patterns will not be used in training\n");
        }
        for(it = 0; it < nPatterns/batchSize; it++){
            //Pick patterns for the batch and compute the gradient

            aggregateGradient(network, AGGRGRAD_ZERO, 0);                     //sets temporary placeholder in network for aggregated gradient to zero. (0 not used)

            for (p = 0; p < batchSize; p++){
                currentIndex = currentBatchesIndexes[it*batchSize + p];

                getArray(patterns, currentPattern, currentIndex, patternSize);  
                getArray(targets, currentTarget, currentIndex, targetSize);


                forwardPropagate(network, currentPattern, currentTarget[1]);      //third argument only used if poisson deviance loss is used
                gradient(network, currentTarget);
                aggregateGradient(network, AGGRGRAD_AGGREGATE, 0);                         // add current pattern related gradient to aggregated gradient. Third argument not used
            }

            aggregateGradient(network, AGGRGRAD_DIVPULL, (double)batchSize);              //when computed gradient for each pattern in the size and aggregated them, divide by batchSize and pull the sum from the temporary variable and put in gradient variable(s)

            t += 1;

            //compute moments updates and then apply updates to weights and biases
            temp = network->next;
            last = 0;
            while(last == 0){                //set velocity to zero before starting to iterate
                if (temp->layerID == 1){
                    nPerceptronsPrevious = network->nPerceptrons;
                }
                else{
                    nPerceptronsPrevious = temp->previous->nPerceptrons;
                }

                for(j = 0; j < temp->nPerceptrons; j++){
                    for (i = 0; i < nPerceptronsPrevious; i++){
                        if (temp->weightStates[i*(temp->nPerceptrons) + j] == 0){
                            //update biased 1st moment estimate
                            temp->wS[i*(temp->nPerceptrons) + j] = rho1*(temp->wS[i*(temp->nPerceptrons) + j]) + (1-rho1)*(temp->weightsGradient[i*(temp->nPerceptrons) + j]);

                            //update biased 2nd moment estimate
                            temp->wR[i*(temp->nPerceptrons) + j] = rho2*(temp->wR[i*(temp->nPerceptrons) + j]) + (1-rho2)*(temp->weightsGradient[i*(temp->nPerceptrons) + j])*(temp->weightsGradient[i*(temp->nPerceptrons) + j]);

                            //correct bias in 1st moment estimate  (THIS IS THE MAIN DIFFERENCE FROM ADAM)
                            hatS = rho1*(temp->wS[i*(temp->nPerceptrons) + j])/(1-pow(rho1,t)) + (1-rho1)*(temp->weightsGradient[i*(temp->nPerceptrons) + j])/(1-pow(rho1,t));

                            //correct bias in 2nd moment estimate (ALSO A MULTIPLICATION HERE WITH rho2 DIFFERS FROM ADAM)
                            hatR = rho2*(temp->wR[i*(temp->nPerceptrons) + j])/(1-pow(rho2,t));

                            //compute update (SMALL DIFFERENCE FROM ADAM, STABILIZATION UNDER SQUARE ROOT)
                            dTheta = -eta*hatS/(sqrt(hatR + epsilon));
                    
                            //apply update
                            temp->weights[i*(temp->nPerceptrons) + j] += dTheta;
                        }
                    } //end for-loop (i) over neurons in previous layer
                    if (temp->biasStates[j] == 0){
                        //update biased 1st moment estimate
                        temp->bS[j] = rho1*(temp->bS[j]) + (1-rho1)*(temp->biasesGradient[j]);

                        //update biased 2nd moment estimate
                        temp->bR[j] = rho2*(temp->bR[j]) + (1-rho2)*(temp->biasesGradient[j])*(temp->biasesGradient[j]);

                        //correct bias in 1st moment estimate (THIS IS THE MAIN DIFFERENCE FROM ADAM)
                        hatS = rho1*(temp->bS[j])/(1-pow(rho1,t)) + (1-rho1)*(temp->biasesGradient[j])/(1-pow(rho1,t));

                        //correct bias in 2nd moment estimate (ALSO A MULTIPLICATION HERE WITH rho2 DIFFERS FROM ADAM)
                        hatR = rho2*(temp->bR[j])/(1-pow(rho2,t));

                        //compute update(SMALL DIFFERENCE FROM ADAM, STABILIZATION UNDER SQUARE ROOT)
                        dTheta = -eta*hatS/(sqrt(hatR + epsilon));

                        //apply update
                        temp->biases[j] += dTheta;
                    }
                } //end for-loop (j) over neurons in current layer
                if (temp->next == NULL){
                    last = 1;
                }
                else{
                    temp = temp->next;
                }
            } //end while-loop over network layers

        } //end iteration-loop

    } //end epoch-loop

    free(currentPattern);
    free(currentTarget);
    free(currentBatchesIndexes);
    return(0);
}
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "nnLib.h"
#include "nnAuxillary.h"

void importData(double *imPatterns, int nPatterns, int linesOffset);
int evaluateNetwork(struct neuralNetwork *network, double *patterns, int patternSize, double *targets, int targetSize, int nSamples, int type);
void getCategory(double *allPatterns, int patternSize, double *catPatterns, int catSize, int offset, int nSamples);
int evaluateCategoryNetwork(struct neuralNetwork *network, double *patterns, int patternSize,double *targets, int targetSize, int nSamples);

int nNeurons = 6;

int batchSize = 1000;
int nEpochs = 500;

double learning = 0.001;




int main(){
    int i;
    int j;
    int k;

    int dataSet;

    int softMax[55];

    int softMaxCat1[6];
    int softMaxCat2[3];
    int softMaxCat3[7];
    int softMaxCat4[11];
    int softMaxCat5[6];
    int softMaxCat6[22];


    double freqCatPatterns[20000*55];  //55 in hot-one coding


   // category sizes {6, 3, 7, 11, 6, 22}
    double freqCatPatternsCat1[20000*6];  
    double freqCatPatternsCat2[20000*3];  
    double freqCatPatternsCat3[20000*7];  
    double freqCatPatternsCat4[20000*11];  
    double freqCatPatternsCat5[20000*6];  
    double freqCatPatternsCat6[20000*22];  


    double feedPatternCat1[6];
    double feedPatternCat2[3];
    double feedPatternCat3[7];
    double feedPatternCat4[11];
    double feedPatternCat5[6];
    double feedPatternCat6[22];

    double tempOutputCat1[6];
    double tempOutputCat2[3];
    double tempOutputCat3[7];
    double tempOutputCat4[11];
    double tempOutputCat5[6];
    double tempOutputCat6[22];


    int corrects = 0;
    int correctLocal = 0;
    int marginal = 0;

    double correctsPercentage = 0.0;
    double marginalPercentage = 0.0;
    
    FILE *fp2;

    struct neuralNetwork autoencoder;

    struct neuralNetwork acCat1;
    struct neuralNetwork acCat2;
    struct neuralNetwork acCat3;
    struct neuralNetwork acCat4;
    struct neuralNetwork acCat5;
    struct neuralNetwork acCat6;

    srand(time(NULL));

/*
    printf("Creating network for each category...\n");

    createNetwork(&acCat1, 6);
    addLayer(&acCat1, 1, actLinear, actLinearPrim);
    addLayer(&acCat1, 6, actLinear, actLinearPrim);

    createNetwork(&acCat2, 3);
    addLayer(&acCat2, 1, actLinear, actLinearPrim);
    addLayer(&acCat2, 3, actLinear, actLinearPrim);

    createNetwork(&acCat3, 7);
    addLayer(&acCat3, 1, actLinear, actLinearPrim);
    addLayer(&acCat3, 7, actLinear, actLinearPrim);

    createNetwork(&acCat4, 11);
    addLayer(&acCat4, 1, actLinear, actLinearPrim);
    addLayer(&acCat4, 11, actLinear, actLinearPrim);

    createNetwork(&acCat5, 6);
    addLayer(&acCat5, 1, actLinear, actLinearPrim);
    addLayer(&acCat5, 6, actLinear, actLinearPrim);

    createNetwork(&acCat6, 6);
    addLayer(&acCat6, 1, actLinear, actLinearPrim);
    addLayer(&acCat6, 6, actLinear, actLinearPrim);
 
    printf("Setting softmax for category networks...\n");


    for(i = 0; i < 6; i++){
        softMaxCat1[i] = 1;
    }
    for(i = 0; i < 3; i++){
        softMaxCat2[i] = 1;
    }
    for(i = 0; i < 7; i++){
        softMaxCat3[i] = 1;
    }
    for(i = 0; i < 11; i++){
        softMaxCat4[i] = 1;
    }
    for(i = 0; i < 6; i++){
        softMaxCat5[i] = 1;
    }
    for(i = 0; i < 22; i++){
        softMaxCat6[i] = 1;
    }
    setSoftMax(&acCat1, softMaxCat1);
    setSoftMax(&acCat2, softMaxCat2);
    setSoftMax(&acCat3, softMaxCat3);
    setSoftMax(&acCat4, softMaxCat4);
    setSoftMax(&acCat5, softMaxCat5);
    setSoftMax(&acCat6, softMaxCat6);
 
    printf("Turning off bias terms in category networks...\n");
    setBiasStates(&acCat1, 1, 1, 1);
    setBiasStates(&acCat2, 1, 1, 1);
    setBiasStates(&acCat3, 1, 1, 1);
    setBiasStates(&acCat4, 1, 1, 1);
    setBiasStates(&acCat5, 1, 1, 1);
    setBiasStates(&acCat6, 1, 1, 1);

    fp2 = fopen("Results.txt","a");
    fprintf(fp2, "Results for %d neurons in hidden layer. Separate. %d epochs. Batch Size: %d. Learning rate: %f.\n", nNeurons, nEpochs, batchSize, learning);
    fclose(fp2);


    for (dataSet = 0; dataSet < 5; dataSet++){

        printf("Importing data set %d...\n", dataSet+1);
        importData(freqCatPatterns,20000,20000*dataSet);
        printf("Done\n\n");


   // category sizes {6, 3, 7, 11, 6, 22}
   // cumulative {0,6,9,16,27,33}
        //separate patterns in order to feed them to different networks
        getCategory(freqCatPatterns, 55, freqCatPatternsCat1, 6, 0, 20000);
        getCategory(freqCatPatterns, 55, freqCatPatternsCat2, 3, 6, 20000);
        getCategory(freqCatPatterns, 55, freqCatPatternsCat3, 7, 9, 20000);
        getCategory(freqCatPatterns, 55, freqCatPatternsCat4, 11, 16, 20000);
        getCategory(freqCatPatterns, 55, freqCatPatternsCat5, 6, 27, 20000);
        getCategory(freqCatPatterns, 55, freqCatPatternsCat6, 22, 33, 20000);


        printf("Initializing weights to uniformly random on [-1,1]...\n");
        setWeights(&acCat1, setToRandom);
        setWeights(&acCat2, setToRandom);
        setWeights(&acCat3, setToRandom);
        setWeights(&acCat4, setToRandom);
        setWeights(&acCat5, setToRandom);
        setWeights(&acCat6, setToRandom);
        setBiases(&acCat1, setToZero);                                
        setBiases(&acCat2, setToZero);                                
        setBiases(&acCat3, setToZero);                                
        setBiases(&acCat4, setToZero);                                
        setBiases(&acCat5, setToZero);                                
        setBiases(&acCat6, setToZero);                                



   // category sizes {6, 3, 7, 11, 6, 22}
        printf("Training network for category 1...\n");
        sgd_nadam(&acCat1, freqCatPatternsCat1, freqCatPatternsCat1, 20000, 6, 6, batchSize, nEpochs, learning, 0.975, 0.999, 0.00000001);
        printf("Done\n");
        printf("Training network for category 2...\n");
        sgd_nadam(&acCat2, freqCatPatternsCat2, freqCatPatternsCat2, 20000, 3, 3, batchSize, nEpochs, learning, 0.975, 0.999, 0.00000001);
        printf("Done\n");
        printf("Training network for category 3...\n");
        sgd_nadam(&acCat3, freqCatPatternsCat3, freqCatPatternsCat3, 20000, 7, 7, batchSize, nEpochs, learning, 0.975, 0.999, 0.00000001);
        printf("Done\n");
        printf("Training network for category 4...\n");
        sgd_nadam(&acCat4, freqCatPatternsCat4, freqCatPatternsCat4, 20000, 11, 11, batchSize, nEpochs, learning, 0.975, 0.999, 0.00000001);
        printf("Done\n");
        printf("Training network for category 5...\n");
        sgd_nadam(&acCat5, freqCatPatternsCat5, freqCatPatternsCat5, 20000, 6, 6, batchSize, nEpochs, learning, 0.975, 0.999, 0.00000001);
        printf("Done\n");
        printf("Training network for category 6...\n");
        sgd_nadam(&acCat6, freqCatPatternsCat6, freqCatPatternsCat6, 20000, 22, 22, batchSize, nEpochs, learning, 0.975, 0.999, 0.00000001);
        printf("Done\n");

      //Evaluate
        
        printf("Evaluating performance...\n");

        corrects = 0;
        marginal = 0;
        for (k = 0; k < 20000; k++){
            correctLocal = 0;

            getArray(freqCatPatternsCat1, feedPatternCat1, k, 6);
            getArray(freqCatPatternsCat2, feedPatternCat2, k, 3);
            getArray(freqCatPatternsCat3, feedPatternCat3, k, 7);
            getArray(freqCatPatternsCat4, feedPatternCat4, k, 11);
            getArray(freqCatPatternsCat5, feedPatternCat5, k, 6);
            getArray(freqCatPatternsCat6, feedPatternCat6, k, 22);


            forwardPropagate(&acCat1, feedPatternCat1);
            forwardPropagate(&acCat2, feedPatternCat2);
            forwardPropagate(&acCat3, feedPatternCat3);
            forwardPropagate(&acCat4, feedPatternCat4);
            forwardPropagate(&acCat5, feedPatternCat5);
            forwardPropagate(&acCat6, feedPatternCat6);

            getOutput(&acCat1, tempOutputCat1);
            getOutput(&acCat2, tempOutputCat2);
            getOutput(&acCat3, tempOutputCat3);
            getOutput(&acCat4, tempOutputCat4);
            getOutput(&acCat5, tempOutputCat5);
            getOutput(&acCat6, tempOutputCat6);



            // category sizes {6, 3, 7, 11, 6, 22}

            i = maxIndex(tempOutputCat1, 6);
            j = maxIndex(feedPatternCat1, 6);
            if (i == j){
                correctLocal += 1;
                marginal += 6;
            }

            i = maxIndex(tempOutputCat2, 3);
            j = maxIndex(feedPatternCat2, 3);
            if (i == j){
                correctLocal += 1;
                marginal += 3;
            }

            i = maxIndex(tempOutputCat3, 7);
            j = maxIndex(feedPatternCat3, 7);
            if (i == j){
                correctLocal += 1;
                marginal += 7;
            }

            i = maxIndex(tempOutputCat4, 11);
            j = maxIndex(feedPatternCat4, 11);
            if (i == j){
                correctLocal += 1;
                marginal += 11;
            }

            i = maxIndex(tempOutputCat5, 6);
            j = maxIndex(feedPatternCat5, 6);
            if (i == j){
                correctLocal += 1;
                marginal += 6;
            }

            i = maxIndex(tempOutputCat6, 22);
            j = maxIndex(feedPatternCat6, 22);
            if (i == j){
                correctLocal += 1;
                marginal += 22;
            }
 
            if (correctLocal == 6){
                corrects += 1;
            }
        }  //end evaluation for loop

        correctsPercentage = corrects/20000.0;
        printf("Data set: %d. Correctly classified (joint precision): %d (%f%%)\n",dataSet+1, corrects, 100*correctsPercentage);

        marginalPercentage = marginal/(55.0*20000.0);
        printf("Data set: %d. Correctly classified (marginal precision): %d (%f%%)\n",dataSet+1, marginal, 100*marginalPercentage);

        fp2 = fopen("Results.txt","a");
        fprintf(fp2, "%d (%f %%), %d (%f%%)\n", corrects, 100*correctsPercentage, marginal, 100*marginalPercentage);
        fclose(fp2);


    } // end data set for loop
*/


    printf("Creating large netwoork...\n");

    createNetwork(&autoencoder, 55);
    addLayer(&autoencoder, nNeurons, actLinear, actLinearPrim);
    addLayer(&autoencoder, 55, actLinear, actLinearPrim);

    printf("Case: softmax_all\n");
//Case softmax_all--------------------------------------------------------------------------------------------------------------------------------------------
    printf("Setting softmax for large network...\n");
    for(i=0; i<55; i++){
/*
        if (i < 6){              
            softMax[i] = 1;
        }
        else if(i < 9){
            softMax[i] = 2;
        }
        else if(i < 16){
            softMax[i] = 3;
        }
        else if(i < 27){
            softMax[i] = 4;
        }
        else if(i < 33){
            softMax[i] = 5;
        }
        else{
            softMax[i] = 6;
        }
*/

        softMax[i] = 1;
    }
    setSoftMax(&autoencoder, softMax);

    printf("Turning of bias terms in first layer...\n\n");
    for (i = 0; i < nNeurons; i++){
        setBiasStates(&autoencoder, 1, i, 1); 
    }
  
    fp2 = fopen("Results.txt","a");
    //fprintf(fp2, "Results for %d neurons in hidden layer. Softmax_per_feat. %d epochs. Batch Size: %d. Learning rate: %f.\n", nNeurons, nEpochs, batchSize, learning);
    fprintf(fp2, "Results for %d neurons in hidden layer. Softmax_all. %d epochs. Batch Size: %d. Learning rate: %f.\n", nNeurons, nEpochs, batchSize, learning);
    fclose(fp2);
    for (dataSet = 0; dataSet < 5; dataSet++){

        printf("Importing data set %d...\n", dataSet+1);
        importData(freqCatPatterns,20000,20000*dataSet);
        printf("Done\n\n");


        printf("Initializing weights to uniformly random on [-1,1]...\n");
        setWeights(&autoencoder, setToRandom); 
        setBiases(&autoencoder, setToZero);                                //ska dessa vara avstängda helt?


        printf("Training network...\n");
        sgd_nadam(&autoencoder, freqCatPatterns, freqCatPatterns, 20000, 55, 55, batchSize, nEpochs, learning, 0.975, 0.999, 0.00000001);
        printf("Done\n");

        printf("Evaluating performance...\n");
  
        corrects = evaluateNetwork(&autoencoder, freqCatPatterns, 55, freqCatPatterns, 55, 20000,0);
        correctsPercentage = corrects/20000.0;
        printf("Data set: %d. Correctly classified (joint precision): %d (%f%%)\n",dataSet+1, corrects, 100*correctsPercentage);

        marginal = evaluateNetwork(&autoencoder, freqCatPatterns, 55, freqCatPatterns, 55, 20000,1);
        marginalPercentage = marginal/(55.0*20000.0);
        printf("Data set: %d. Correctly classified (marginal precision): %d (%f%%)\n",dataSet+1, marginal, 100*marginalPercentage);

        fp2 = fopen("Results.txt","a");
        fprintf(fp2, "%d (%f %%), %d (%f%%)\n", corrects, 100*correctsPercentage, marginal, 100*marginalPercentage);
        fclose(fp2);
    }



//Case softmax_per_feature--------------------------------------------------------------------------------------------------------------------------------------------

    printf("Case: softmax_per_feature\n");
    printf("Setting softmax for large network...\n");
    for(i=0; i<55; i++){

        if (i < 6){              
            softMax[i] = 1;
        }
        else if(i < 9){
            softMax[i] = 2;
        }
        else if(i < 16){
            softMax[i] = 3;
        }
        else if(i < 27){
            softMax[i] = 4;
        }
        else if(i < 33){
            softMax[i] = 5;
        }
        else{
            softMax[i] = 6;
        }

  //      softMax[i] = 1;
    }
    setSoftMax(&autoencoder, softMax);

    printf("Turning of bias terms in first layer...\n\n");
    for (i = 0; i < nNeurons; i++){
        setBiasStates(&autoencoder, 1, i, 1); 
    }
  
    fp2 = fopen("Results.txt","a");
    fprintf(fp2, "Results for %d neurons in hidden layer. Softmax_per_feat. %d epochs. Batch Size: %d. Learning rate: %f.\n", nNeurons, nEpochs, batchSize, learning);
    //fprintf(fp2, "Results for %d neurons in hidden layer. Softmax_all. %d epochs. Batch Size: %d. Learning rate: %f.\n", nNeurons, nEpochs, batchSize, learning);
    fclose(fp2);
    for (dataSet = 0; dataSet < 5; dataSet++){

        printf("Importing data set %d...\n", dataSet+1);
        importData(freqCatPatterns,20000,20000*dataSet);
        printf("Done\n\n");


        printf("Initializing weights to uniformly random on [-1,1]...\n");
        setWeights(&autoencoder, setToRandom); 
        setBiases(&autoencoder, setToZero);                                //ska dessa vara avstängda helt?


        printf("Training network...\n");
        sgd_nadam(&autoencoder, freqCatPatterns, freqCatPatterns, 20000, 55, 55, batchSize, nEpochs, learning, 0.975, 0.999, 0.00000001);
        printf("Done\n");

        printf("Evaluating performance...\n");
  
        corrects = evaluateNetwork(&autoencoder, freqCatPatterns, 55, freqCatPatterns, 55, 20000,0);
        correctsPercentage = corrects/20000.0;
        printf("Data set: %d. Correctly classified (joint precision): %d (%f%%)\n",dataSet+1, corrects, 100*correctsPercentage);

        marginal = evaluateNetwork(&autoencoder, freqCatPatterns, 55, freqCatPatterns, 55, 20000,1);
        marginalPercentage = marginal/(55.0*20000.0);
        printf("Data set: %d. Correctly classified (marginal precision): %d (%f%%)\n",dataSet+1, marginal, 100*marginalPercentage);

        fp2 = fopen("Results.txt","a");
        fprintf(fp2, "%d (%f %%), %d (%f%%)\n", corrects, 100*correctsPercentage, marginal, 100*marginalPercentage);
        fclose(fp2);
    }


    //viewNetwork(&autoencoder);

    return(0);
}



void importData(double *imPatterns, int nPatterns, int linesOffset){
  FILE *fp;
  char buffer[10];
  int i;
  int j;
  int k;


  int categorySizes[6] = {6, 3, 7, 11, 6, 22};
  int cumCategories[6] = {0,6,9,16,27,33};

  int totSize = 55;
  //int nPatterns = 200;

  char tmp = 0;

  fp = fopen("kat.csv","r");

  //skip first row
  for(i=0; i<100; i++){
    tmp =fgetc(fp);
    if (tmp == '\n'){
      break;
    }
  }

  //Skip offset
  for (j = 0; j < linesOffset; j++){
    for(i=0; i<100; i++){
      tmp =fgetc(fp);
      if (tmp == '\n'){
        break;
      }
    }
  }

  for (k = 0 ; k < nPatterns; k++){  
    for (i = 0; i < totSize; i++){
      imPatterns[k*totSize + i] = 0.0;
    }
    //skipp first column
    for (i = 0; i < 30; i++){
      tmp = fgetc(fp);
      if (tmp == ','){
        break;
      }
    }
  
    //second column and so on, save them.
    for (j = 0; j < 6; j++){
      for (i = 0; i < 10; i++){
          buffer[i] = 0;
      }
      for (i = 0; i < 30; i++){
        tmp = fgetc(fp);
        if ((tmp == ',')||(tmp == '\n')){
          break;
        }
        buffer[i] = tmp;
      }
      //printf("%d,", atoi(buffer));
      imPatterns[k*totSize + cumCategories[j] + atoi(buffer)] = 1.0;
    }
    //printf("\n");
    
//    for (i = 0; i < totSize; i++){
//       for (j = 0; j < 6; j++){
//           if (i == cumCategories[j]){
//               printf("\n\n");
//           }
//       }
//       printf("%f, ", imPatterns[k*totSize + i]);
//    }
//    printf("\n\n");
  }

  fclose(fp);
}



int evaluateNetwork(struct neuralNetwork *network, double *patterns, int patternSize,
                                                   double *targets, int targetSize,
                                                   int nSamples, int type){
    int corrects;
    int correctLocal;
    int k;
    int i;
    int j;

    //type: 0 - join, 1 - marginal

    double *feedzPattern;
    double *feedzTarget;
    double *tempzOutput;

    feedzPattern = malloc(patternSize*sizeof(double));
    feedzTarget = malloc(targetSize*sizeof(double));
    tempzOutput = malloc(targetSize*sizeof(double));

    corrects = 0;
    for (k = 0; k < nSamples; k++){
        getArray(patterns, feedzPattern, k, patternSize);
        getArray(targets, feedzTarget, k, targetSize);

        forwardPropagate(network, feedzPattern);
        getOutput(network, tempzOutput);

        correctLocal = 0;

        i = maxIndexBetween(tempzOutput, 55,0,5);
        j = maxIndexBetween(feedzTarget, 55,0,5);
        if (i == j){
            correctLocal += 1;
            if (type == 1){
                corrects += 6;
            }
        }

        i = maxIndexBetween(tempzOutput, 55,6,8);
        j = maxIndexBetween(feedzTarget, 55,6,8);
        if (i == j){
            correctLocal += 1;
            if (type == 1){
                corrects += 3;
            }
        }

        i = maxIndexBetween(tempzOutput, 55,9,15);
        j = maxIndexBetween(feedzTarget, 55,9,15);
        if (i == j){
            correctLocal += 1;
            if (type == 1){
                corrects += 7;
            }
        }

        i = maxIndexBetween(tempzOutput, 55,16,26);
        j = maxIndexBetween(feedzTarget, 55,16,26);
        if (i == j){
            correctLocal += 1;
            if (type == 1){
                corrects += 11;
            }
        }

        i = maxIndexBetween(tempzOutput, 55,27,32);
        j = maxIndexBetween(feedzTarget, 55,27,32);
        if (i == j){
            correctLocal += 1;
            if (type == 1){
                corrects += 6;
            }
        }

        i = maxIndexBetween(tempzOutput, 55,33,55);
        j = maxIndexBetween(feedzTarget, 55,33,55);
        if (i == j){
            correctLocal += 1;
            if (type == 1){
                corrects += 22;
            }
        }

        if (type == 0){
            if (correctLocal == 6){
                corrects += 1;
            }
        }
    }    

    free(feedzPattern);
    free(feedzTarget);
    free(tempzOutput);

    return(corrects);
}

void getCategory(double *allPatterns, int patternSize, double *catPatterns, int catSize, int offset, int nSamples){
    int i;
    int k;
    for(k = 0; k < nSamples; k++){
        for(i = 0; i < catSize; i++){
            catPatterns[k*catSize + i] = allPatterns[k*patternSize + offset + i];
        }
    }
}

int evaluateCategoryNetwork(struct neuralNetwork *network, double *patterns, int patternSize,
                                                   double *targets, int targetSize,
                                                   int nSamples){
    int corrects;

    int k;
    int i;
    int j;

    double *feedzPattern;
    double *feedzTarget;
    double *tempzOutput;

    feedzPattern = malloc(patternSize*sizeof(double));
    feedzTarget = malloc(targetSize*sizeof(double));
    tempzOutput = malloc(targetSize*sizeof(double));

    corrects = 0;
    for (k = 0; k < nSamples; k++){
        getArray(patterns, feedzPattern, k, patternSize);
        getArray(targets, feedzTarget, k, targetSize);

        forwardPropagate(network, feedzPattern);
        getOutput(network, tempzOutput);


        i = maxIndex(tempzOutput, targetSize);
        j = maxIndex(feedzTarget, targetSize);
        if (i == j){
            corrects += 1;
        }        
    }    

    free(feedzPattern);
    free(feedzTarget);
    free(tempzOutput);

    return(corrects);
}
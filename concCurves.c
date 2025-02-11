#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N_SAMPLES 20000

double concentrationCurve(double alpha, double *yTrue, double *yPred){
  double resultValue = 0.0;
  double yTotSum = 0.0;
  int n = N_SAMPLES;
  int i;
  int j;
  double partSum = 0.0;


  for (i=0; i < n; i++){
    yTotSum += yTrue[i];
  }

  partSum = 0.0;
  for (i=0; i<n; i++){
    partSum = 0.0;
    for (j=0; j<n; j++){
      if (yPred[j] <= yPred[i]){
        partSum += 1;
      }
    }
    partSum = partSum/n;
    if (partSum <= alpha){
      resultValue += yTrue[i];
    }
  }
  resultValue = resultValue/yTotSum;
  return(resultValue);
}

int main(){
  FILE *fp;
  FILE *fp2;

  double y_true[N_SAMPLES];
  double y_pred[N_SAMPLES];
  int i;
  int j;
  char buffer[10];
  float bufferF = 0.0;
  double bufferD = 0.0;

  double theResult = 0.0;
  double concCurve[10];

  fp = fopen("y_pred.txt","r");
  fp2 = fopen("y_true.txt","r");

  for (i=0; i< N_SAMPLES; i++){
      fscanf(fp, "%f",&bufferF);
      y_pred[i] = bufferF;
      fscanf(fp2, "%f",&bufferF);
      y_true[i] = bufferF;
      
  }
  printf("Values imported!\n");
  fclose(fp);
  fclose(fp2);

  fp = fopen("cc.txt", "w");

  for (i=0; i<11; i++){
     bufferD = 0.1*i;
     theResult = concentrationCurve(bufferD, y_true, y_pred);
     printf("alpha = %f ---> concentration curve = %f\n", bufferD, theResult);
     fprintf(fp, "%f\n", theResult);
  }

  fclose(fp);
  return(0);
}
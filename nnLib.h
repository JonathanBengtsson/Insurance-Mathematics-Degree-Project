#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>


#define NPARTIALRESULTS 1

#define AGGRGRAD_AGGREGATE 1
#define AGGRGRAD_ZERO 2
#define AGGRGRAD_PULL 3
#define AGGRGRAD_DIVIDE 4
#define AGGRGRAD_DIVPULL 5


struct neuralNetworkLayer{
  double *perceptrons;
  double *weights;
  double *biases;

  double *partialResults;  //for gradient computation
  double *inSignal;       //signal in for each perceptron before activation function. Speeds up computations

  double *weightsGradient;
  double *weightsAggrGradient;   //temporary, when using batches
  double *biasesGradient;
  double *biasesAggrGradient;    //temporary, when using batches

  double *weightsOld;    //used for nesterov
  double *biasesOld;

  double *wVelocity;     //used for nesterov
  double *bVelocity;

  double *wR;            //used for moments in ADAM and NADAM
  double *bR;
  double *wS;
  double *bS;

  //double *dropedOutPerceptronsValue;
  //double *dropedOutWeightsValue;
  //int *dropedOutPerceptronsID;

  char *weightStates;
  char *biasStates;
  char *isInput;

  int *softMax;

  int nPerceptrons;
  int layerID;
  int lastLayer;

  //int dropOutActive;

  double (*activationFunction)(double);
  double (*derrivActivationFunction)(double);

  struct neuralNetworkLayer *next;
  struct neuralNetworkLayer *previous;

};


struct neuralNetwork{
  double *perceptrons;
  int nPerceptrons;
  int layerID;

  char poissonLoss;

  struct neuralNetworkLayer *next;
};

struct neuralNetworkLayer * getLastLayer(struct neuralNetwork *network);
void createNetwork(struct neuralNetwork *network, int v1);
int addLayer(struct neuralNetwork *network, int v1, double (*v2)(double), double (*v3)(double));
void viewNetwork(struct neuralNetwork *network);
int viewOutput(struct neuralNetwork *network);
int getOutput(struct neuralNetwork *network, double *output);
void setWeights(struct neuralNetwork *network, double (*setToThis)(void));
void setWeightsRandom(struct neuralNetwork *network);
void setBiases(struct neuralNetwork *network, double (*setToThis)(void));
void setThresholdsRandom(struct neuralNetwork *network);
int forwardPropagate(struct neuralNetwork *network, double pattern[], double exposure);
int backwardPropagate_SGD(struct neuralNetwork *network, double targetPattern[], double eta);
int gradient(struct neuralNetwork *network, double targetPattern[]);

int stochastic_gradient_descent(struct neuralNetwork *network, double patterns[], double targets[], int nPatterns, int patternSize, int targetSize, int nEpochs, double eta);
int sgd_nestrov(struct neuralNetwork *network, double patterns[], double targets[], int nPatterns, int patternSize, int targetSize, int nEpochs, double eta, double alpha);
int sgd_adam(struct neuralNetwork *network, double patterns[], double targets[], int nPatterns, int patternSize, int targetSize, int nEpochs, double eta, double rho1, double rho2, double delta);
int sgd_nadam(struct neuralNetwork *network, double patterns[], double targets[], int nPatterns, int patternSize, int targetSize, int batchSize, int nEpochs, double eta, double rho1, double rho2, double epsilon);

int setSoftMax(struct neuralNetwork *network, int *softMaxIDs);

//int setDropOutIDs(struct neuralNetwork *network, int layer, int dropOutIDs[]);
//int dropOutOn(struct neuralNetwork *network);
//int dropOutOff(struct neuralNetwork *network);

int setWeightStates(struct neuralNetwork *network, int layer, int connectionThis, int connectionPrevious, int setTo);
int setBiasStates(struct neuralNetwork *network, int layer, int neuron, int setTo);
int setInput(struct neuralNetwork *network, int layer, int neuron);
int setPoissonLoss(struct neuralNetwork *network);

void aggregateGradient(struct neuralNetwork *network, int action, double operand);

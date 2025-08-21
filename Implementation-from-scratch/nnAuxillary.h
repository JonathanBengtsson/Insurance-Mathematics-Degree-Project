double setToRandom(void);
double setToZero(void);
double setToOne(void);

double actLinear(double input);
double actLinearPrim(double input);
double actSigmoid(double input);
double actSigmoidPrim(double input);
double actTanh(double input);
double actTanhPrim(double input);

void getArray(double patterns[], double outputPattern[], int patternID, int patternSize);
void generateOrder(int indexes[], int nIndex);

int arrMax_int(int *arr, int arrLength);
int maxIndex(double *arr, int arrLength);
int maxIndexBetween(double *arr, int arrLength, int smallest, int largest);

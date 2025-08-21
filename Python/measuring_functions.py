import keras.backend as K


def poisson_deviance_loss(y_true, y_pred):
    y_pred = K.maximum(y_pred, 0.0 + K.epsilon())                               #make sure ypred is positive or ln(-x) = NAN
    y_true = K.maximum(y_true, 0.0 + K.epsilon())                               #make sure ypred is positive or ln(-x) = NAN

    return 2*K.mean(y_true*K.log(y_true/y_pred)-(y_true-y_pred))


def devianceBis(y_true, y_pred):
    y_pred = K.maximum(y_pred, 0.0 + K.epsilon())                               #make sure ypred is positive or ln(-x) = NAN
    return (K.sqrt(K.square( 2 * K.log(y_true + K.epsilon()) - K.log(y_pred))))


def linearAct(x_arg):
    return(x_arg)

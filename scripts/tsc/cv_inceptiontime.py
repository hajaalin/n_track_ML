import click
from datetime import datetime
import logging
import numpy as np
from pathlib import Path
import pandas as pd
from scikeras.wrappers import KerasClassifier
import shap
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder
import sys
import tensorflow as tf
import time
# https://stackoverflow.com/questions/66814523/shap-deepexplainer-with-tensorflow-2-4-error
#from tensorflow.compat.v1.keras.backend import get_session
#tf.compat.v1.disable_v2_behavior() 

from utility import parse_config
from load_data_tsc import load_data, fsets


logger = logging.getLogger(__name__)


def get_standard_scaling(X):
    print("get_standard_scaling")
    print(X.shape)
    # X dimensions:
    # i: samples, ~200 series
    # j: the time series, ~30 steps
    # k: features, ~3-10 features
    mean = np.mean(X,axis=(0,1))
    std = np.std(X, axis=(0,1))

    print(mean.shape)
    print(std.shape)
    
    return mean,std

def apply_standard_scaling(X,mean,std):
    print("apply_standard_scaling")
    print(X.shape)
    # X dimensions:
    # i: samples, ~200 series
    # j: the time series, ~30 steps
    # k: features, ~3-10 features

    print(mean.shape)
    print(std.shape)
    # trust numpy broadcasting
    Xt = X - mean
    Xt = Xt / std

    print(Xt.shape)

    return Xt


## Look for a high-performing model, they explain why the average is so high.
## But what is it about that split that makes it so successful?
TARGET_ACCURACY = 0.98


def get_shap_values(model_, X_train, X_test):
    # SHAP
    shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
    explainer = shap.DeepExplainer((model_.layers[0].input, \
                                    model_.layers[-1].output), \
                                   X_train)
    shap_values_deep = explainer.shap_values(X_test)

    explainer = shap.GradientExplainer(model_, X_train)
    #explainer.expected_value = explainer.expected_value[0]
    shap_values_grad = explainer.shap_values(X_test)

    return (shap_values_deep, shap_values_grad)
   


'''
Repeat cross-validation
'''
def inceptiontime_cv_repeat(X, y, groups, features, output_it, fset, kernel_size=20, epochs=250, use_bottleneck=True, repeats=1,job_id='', save_shap_values=False, set_split_random_state=False, verbose=False):
    logger.info(fset)
    logger.debug("X: " + str(X.shape))
    logger.debug("y: " + str(y.shape))

    # fset includes also class and file, here they have been removed
    print(features)

    nb_classes = 2
    input_shape = X.shape[1:]
    batch_size = int(min(X.shape[0] / 10, 16))
    #verbose = False
    print("input_shape")
    print(input_shape)

    
    # save original y 
    y_true = y.astype(np.int64)
    
    # transform the labels from integers to one hot vectors
    # https://github.com/hajaalin/InceptionTime/blob/f3fd6c5e9298ec9ca5d0fc594bb07dd1decc3718/main.py#L15
    enc = OneHotEncoder()
    enc.fit(np.concatenate((y,), axis=0).reshape(-1, 1))
    y = enc.transform(y.reshape(-1, 1)).toarray()

    
    # columns recorded for each round of cross-validation
    columns = ['accuracy','precision','recall','f1','repeat']
    scores = pd.DataFrame(columns=columns)

    # containers for saving data for model evaluation
    list_idx_train = []
    list_idx_test = []
    list_accuracy = []
    list_shap_deep = []
    list_shap_grad = []
    list_X_test = []

    for i in range(repeats):
        if set_split_random_state:
            cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
        else:
            cv = StratifiedGroupKFold(n_splits=4, shuffle=True)
        print(cv)
        
        print('repeat: %d/%d' % (i+1, repeats))
        logger.debug('repeat: %d/%d' % (i+1, repeats))

        # This import will fail if placed in the beginning of the file.
        # InceptionTime source directory must be added to sys.path first.
        from classifiers import inception
        def create_model():
            clsfr = inception.Classifier_INCEPTION(output_it, \
                                                   input_shape, \
                                                   nb_classes, \
                                                   batch_size=batch_size, \
                                                   use_bottleneck=use_bottleneck, \
                                                   kernel_size=kernel_size, \
                                                   nb_epochs=epochs, \
                                                   verbose=verbose)
            #print(clsfr.model)
            #clsfr.model.summary()
            return clsfr.model


        # One-hot encoding is a problem for StratifiedGroupKFold,
        # split using y_true
        for train_index,val_index in cv.split(X,y_true,groups):
            #print('cv loop')

            #print(train_index)
            #print(val_index)
            #continue
            #print(y_true[train_index])
            #print(y_inc[train_index])
            #break

            # scale training data to mean,std 0,1
            mean,std = get_standard_scaling(X[train_index])
            logger.debug("mean:")
            logger.debug(mean)
            logger.debug("std:")
            logger.debug(std)
            X_train_scaled = apply_standard_scaling(X[train_index],mean,std)
            print("check scaled shape")
            print(X[train_index].shape)
            print(X_train_scaled.shape)
            #logger.debug("scaled2")
            #logger.debug(X_train_scaled)

            classifier = KerasClassifier(model=create_model(), \
                                         epochs=epochs, \
                                         batch_size=batch_size, \
                                         verbose=verbose)
            classifier.fit(X_train_scaled, y[train_index])

            # scale validation data to mean,std 0,1
            #X_val_scaled = standard_scale_x_by_series(X_inc[val_index])
            X_val_scaled = apply_standard_scaling(X[val_index],mean,std)
            pred = classifier.predict(X_val_scaled)

            truth = y_true[val_index]

            #print('truth')
            #print(truth)
            #print('pred')
            #print(pred)

            # prediction is onehot-encoded, reverse it
            pred = pred.argmax(1)
            #print(pred)

            # get fold accuracy and append to dataframe
            fold_acc = accuracy_score(truth, pred)
            fold_prc = precision_score(truth, pred)
            fold_rec = recall_score(truth, pred)
            fold_f1 = f1_score(truth, pred)
            scores.loc[len(scores)] = [fold_acc,fold_prc,fold_rec,fold_f1,i+1]

            print("fold_acc %f0.00" % fold_acc)

            list_accuracy.append(fold_acc)
            list_idx_train.append(train_index)
            list_idx_test.append(val_index)
            if save_shap_values:
                (shap_deep, shap_grad) = get_shap_values(classifier.model_, \
                                                         X_train_scaled, \
                                                         X_val_scaled)
                list_shap_deep.append(shap_deep[1])
                list_shap_grad.append(shap_grad[1])
                list_X_test.append(X_val_scaled)

        print("len(list_shap_deep):%d" % len(list_shap_deep))
        print("len(list_shap_grad):%d" % len(list_shap_grad))
        
    # once accuracy metrics have been recorded,
    # add columns with metadata
    scores['cv'] = str(cv)
    scores['classifier'] = 'InceptionTime'
    scores['fset'] = fset
    scores['kernel_size'] = kernel_size
    scores['epochs'] = epochs
    scores['job_id'] = job_id


    logger.info(f"inceptiontime_cv_repeat: %s feature_set:%s, kernel_size:%d, epochs:%d, accuracy:%f0.00" % (str(cv), fset, kernel_size, epochs, scores['accuracy'].mean()))

    lists_all = (list_accuracy, list_idx_train, list_idx_test, \
                 list_shap_deep, list_shap_grad, list_X_test)

    return scores, lists_all


def shap2npy(fset, shap_lists_all, output_shap):
    (list_accuracy, list_idx_train, list_idx_test, \
     list_shap_deep, list_shap_grad, list_X_test) = shap_lists_all

    np.save(output_shap / 'list_idx_train.npy', list_idx_train)
    np.save(output_shap / 'list_idx_test.npy', list_idx_test)
    np.save(output_shap / 'list_accuracy.npy', list_accuracy)
    np.save(output_shap / 'list_shap_deep.npy', list_shap_deep)
    np.save(output_shap / 'list_shap_grad.npy', list_shap_grad)
    np.save(output_shap / 'list_X_test.npy', list_X_test)
    np.save(output_shap / 'features.npy', fsets[fset])
        


@click.command()
@click.option("--inceptiontime_dir", type=str)
@click.option("--paths", type=str, default="paths.yml")
@click.option("--use_bottleneck", is_flag=True, default=True)
@click.option("--kernel_size", type=int, default=20)
@click.option("--epochs", type=int, default=100)
@click.option("--fset", type=click.Choice(fsets.keys()), default="f_mot_morph")
@click.option("--repeats", type=int, default=20)
@click.option("--save_shap_values", is_flag=True, default=False)
@click.option("--job_name", type=str, default="tsc_it")
@click.option("--job_id", type=str)
@click.option("--now", type=str)
def cv_inceptiontime(inceptiontime_dir, paths, use_bottleneck, kernel_size, epochs, fset, repeats, save_shap_values, job_name, job_id, now):
    paths = parse_config(paths)

    log_dir = Path(paths["log"]["tsc"]) / job_name / now
    log_dir.mkdir(parents=True, exist_ok=True)

    if not now:
        now = datetime.now().strftime("%Y%m%d%H%M%S")

    # configure logger
    log_file = log_dir / (f"cv_inceptiontime_%s_%s_%s.log" % (job_name, now, job_id))
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_file, mode="w")
    formatter = logging.Formatter(
        "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f"Finished logger configuration!")
    print("Logging to " + str(log_file))

    # log the version of this code
    logger.info(Path(__file__).absolute())

    # add InceptionTime source to Python path
    if inceptiontime_dir == 'TEST':
        # this is test mode, without cloning code from GitHub
        inceptiontime_dir = paths["src"]["inceptiontime"]
    sys.path.insert(1, inceptiontime_dir)

    # log the InceptionTime version
    logger.info(inceptiontime_dir)

    # output folders
    output_cv = Path(paths["output"]["cv"]) / job_name / now
    output_cv.mkdir(parents=True, exist_ok=True)
    output_it = Path(paths["output"]["it"]) / job_id
    output_it.mkdir(parents=True, exist_ok=True)
    output_it = str(output_it) + "/"
    output_shap = Path(paths["output"]["shap"]) / job_name / now / job_id
    output_shap.mkdir(parents=True, exist_ok=True)

    # read the data 
    data_dir = paths["data"]["dir"]
    raw_data_file = paths["data"]["raw_data_file"]
    raw_data_file = Path(data_dir) / raw_data_file

    X, y, groups, features = load_data(raw_data_file, fset)

    tic = time.perf_counter()
    
    logger.info("Start processing...")
    scores, shap_lists_all = inceptiontime_cv_repeat(X, y, groups, features, output_it, fset, use_bottleneck=use_bottleneck, kernel_size=kernel_size, epochs=epochs, repeats=repeats, save_shap_values=save_shap_values, job_id=job_id)
        
    toc = time.perf_counter()
    logger.info(f"Finished processing in {(toc-tic) / 60:0.1f} minutes.")
            
    scores_file = "cv_" + job_name + "_k" + str(kernel_size) + "_e" + str(epochs) + "_" + now + ".csv"
    scores_file = output_cv / scores_file
    scores.to_csv(scores_file, index=False)
    logger.info("Wrote scores to " + str(scores_file))


    # save X_test and accuracy lists, even if SHAP values are not calculated
    shap2npy(fset, shap_lists_all, output_shap)

        

if __name__ == "__main__":
    cv_inceptiontime()
    


from IPython.core.interactiveshell import InteractiveShell
from itertools import product
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import tensorflow as tf

def plot_confusion_matrix(predictions, labels, class_names=None, cmap='inferno'):
    '''
    This is a modified version of the the sci-kit learn function plot_confusion_matrix
    so that we can use it without a sci-kit-learn estimator
    '''
    fig, ax = plt.subplots(figsize=(8, 8))

    cm = confusion_matrix(labels, predictions)
    n_classes = cm.shape[0]
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    cmap_min, cmap_max = im.cmap(0), im.cmap(256)

    text = np.empty_like(cm, dtype=object)

    # print text with appropriate color depending on background
    thresh = (cm.max() + cm.min()) / 2.0

    for i, j in product(range(n_classes), range(n_classes)):
        color = cmap_max if cm[i, j] < thresh else cmap_min

        text_cm = format(cm[i, j], '.2g')
        if cm.dtype.kind != 'f':
            text_d = format(cm[i, j], 'd')
            if len(text_d) < len(text_cm):
                text_cm = text_d

        text[i, j] = ax.text(
            j, i, text_cm,
            ha="center", va="center",
            color=color)

    if class_names:
        display_labels = class_names
    else:
        display_labels = np.arange(n_classes)

    fig.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(n_classes),
            yticks=np.arange(n_classes),
            xticklabels=display_labels,
            yticklabels=display_labels,
            ylabel="True label",
            xlabel="Predicted label")
    plt.title('Confusion Matrix')
    plt.show()

def plot_training_metrics(history):
    '''
    Takes a history object generated along with model training and plots
    the included training metrics. This assumes the model was trained
    with validation data.
    '''
    metrics = list(history.history.keys())
    num_metrics = len(metrics)//2
    fig = plt.figure(figsize=(12, 4*num_metrics))
    for i in range(num_metrics):
        metric = metrics[i]
        cap_metric = metric.capitalize()
        val_metric = 'val_' + metric
        plt.subplot(math.ceil(num_metrics/2), 2, i+1)
        plt.plot(history.history[metric], label=f'Training {cap_metric}')
        plt.plot(history.history[val_metric], label=f'Validation {cap_metric}')
        plt.xlabel('Epochs')
        plt.ylabel(cap_metric)
        plt.title(cap_metric)
        plt.legend()
    plt.suptitle('Model Training')
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

def save_best_model(model, filename, test_x, test_y=None, save_traces=True):
    '''
    Compares the performance of a saved model stored at filename with the provided model on
    the provided test dataset. If the provided model performs better than the saved model
    the provided model is saved to filename, overwriting the previously saved model. If
    no saved model is present at filename then the provided model is saved to filename
    '''
    try:
        saved_model = tf.keras.models.load_model(filename)
    except:
        saved_model = None

    if saved_model:
        print('Saved Model Evaluation:')
        saved_model_metrics = saved_model.evaluate(x=test_x, y=test_y, return_dict=True)
        print('New Model Evaluation:')
        new_model_metrics = model.evaluate(x=test_x, y=test_y, return_dict=True)
        if new_model_metrics['loss'] < saved_model_metrics['loss']:
            model.save(filename, save_traces=save_traces)
    else:
        print('New Model Evaluation:')
        new_model_metrics = model.evaluate(x=test_x, y=test_y)
        model.save(filename, save_traces=save_traces)

def show_all_jupyter_output():
    '''
    Changes the Jupyter notebook's settings to show all output instead of just 
    the last output
    '''
    InteractiveShell.ast_node_interactivity = "all"

def write_vocab_file(filepath, vocab):
    with open('vocab\\'+filepath, 'w') as f:
        for token in vocab:
            print(token, file=f)
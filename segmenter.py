# Diego Alberto Barriga Martinez @umoqnier
# Replica del experimento  'Automatic Prediction of Lezgi Morpheme Breaks'
# realizado para la lengua Lezgi para generar glosa automática a partir
# de escasos ejemplos

import click
import json
import numpy as np
from sklearn.model_selection import KFold
from utils import ( WordsToLetter, sent2features, sent2labels, extractTokens,
                   extractLabels, extractFeatures, eval_labeled_positions,
                   bio_classification_report, labels_decoder,
                   print_state_features, print_transitions, accuracy_score,
                   write_report, get_train_test, param_setter, XMLtoWords,
                   model_trainer, model_tester, get_corpus)


@click.command()
@click.option('--corpora-path', default='corpora/', type=click.Path(),
              help='Carpeta donde se encuentra el corpus para entrenamiento')
@click.option('--models-path', default='models/', type=click.Path(),
              help='Carpeta donde se guardan los modelos entrenados')
@click.option('-n', '--model-name', type=str,
              help='Nombre del modelo a entrenar')
@click.option('-t', '--test-size', type=float,
              help='Tamaño del conjunto de pruebas')
@click.option('-i', '--iterations', type=int,
              help='Número máximo de iteraciones de entrenamiento')
@click.option('-l1', type=float,
              help='Parametro de optimización ElasticNet L1')
@click.option('-l2', type=float,
              help='Parametro de optimización ElasticNet L2')
@click.option('-e', '--evaluation', type=str,
              help='Método de evaluación. Ej. hold_out, k_fold')
@click.option('-d', '--debug', is_flag=True,
              help='Habilita el modo depuración')
@click.option('-v', '--verbose', is_flag=True,
              help='Habilita el modo verboso')
@click.argument('hparams', type=click.File('r'))
def cli(corpora_path, models_path, model_name, debug, verbose, test_size,
        iterations, l1, l2, evaluation, hparams):
    """Command Line Interface para glosador automático del otomí (hñahñu)
    """
    if debug:
        breakpoint()
    hyper = json.loads(hparams.read())
    hyper = param_setter(hyper, model_name, test_size, iterations, l1, l2,
                         evaluation)
    if hyper['dataset-train'] == "lezgi":
        corpus = XMLtoWords('FLExTxtExport2.xml')
    else:
        corpus = get_corpus(hyper['dataset-train'], corpora_path)
    if hyper['evaluation'] == 'hold_out':
        print("*"*10)
        print("HOLD OUT VALIDATION")
        print("*"*10)
        train_data, test_data = get_train_test(corpus, hyper['test-split'],
                                               hyper['dataset-test'],
                                               corpora_path)
        train_size = len(train_data)
        test_size = len(test_data)
        train_time, compositive_name = model_trainer(train_data, models_path,
                                                     hyper, verbose)
        y_test, y_pred, tagger = model_tester(test_data, models_path,
                                              compositive_name, verbose)
        accuracy = accuracy_score(y_test, y_pred)
        train_time_format = str(round(train_time / 60, 2)) + "[m]"
        write_report(compositive_name, train_size, test_size, accuracy,
                     train_time_format, hyper)
        if verbose:
            eval_labeled_positions(y_test, y_pred)
            print(bio_classification_report(y_test, y_pred))
    else:
        i = 0
        partial_time = 0
        partial_accuracy = 0
        kf = KFold(n_splits=hyper['k'], shuffle=True)
        hard_corpus = get_corpus('corpus_hard', corpora_path)
        corpus = WordsToLetter(corpus)
        hard_corpus = WordsToLetter(hard_corpus)
        dataset = np.array(corpus + hard_corpus
        print("*"*10)
        print("K FOLDS VALIDATION")
        print("*"*10)
        for train_index, test_index in kf.split(dataset):
            i += 1
            print("Iteration #", i)
            train_data, test_data = dataset[train_index], dataset[test_index]
            train_time, compositive_name = model_trainer(train_data,
                                                         models_path, hyper,
                                                         verbose, i)
            y_test, y_pred, tagger = model_tester(test_data, models_path,
                                                  compositive_name, verbose)
            partial_accuracy += accuracy_score(y_test, y_pred)
            partial_time += train_time
            if verbose:
                print("*"*10)
                print("Partial Time>>", train_time, "Accuracy acumulado>>",
                      partial_accuracy)
                eval_labeled_positions(y_test, y_pred)
                print(bio_classification_report(y_test, y_pred))
        accuracy = partial_accuracy / hyper['k']
        train_time_format = str(round(partial_time / 60, 2)) + "[m]"
        train_size = len(train_data)
        test_size = len(test_data)
        print("Time>>", train_time_format, "Accuracy>>", accuracy)
        write_report(compositive_name, train_size, test_size, accuracy,
                     train_time_format, hyper)


if __name__ == '__main__':
    cli()

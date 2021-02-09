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
@click.option('-i', '--iterations', type=int,
              help='Número máximo de iteraciones de entrenamiento')
@click.option('-l1', type=float,
              help='Parametro de optimización ElasticNet L1')
@click.option('-l2', type=float,
              help='Parametro de optimización ElasticNet L2')
@click.option('-d', '--debug', is_flag=True,
              help='Habilita el modo depuración')
@click.option('-v', '--verbose', is_flag=True,
              help='Habilita el modo verboso')
@click.argument('hparams', type=click.File('r'))
def cli(corpora_path, models_path, model_name, debug, verbose, iterations,
        l1, l2, hparams):
    """Command Line Interface para glosador automático del otomí (hñahñu)
    """
    if debug:
        breakpoint()
    params_set = json.loads(hparams.read())
    for params in params_set:
        hyper = param_setter(params, model_name, iterations, l1, l2)
        if hyper['dataset'] == "lezgi":
            corpus = XMLtoWords('FLExTxtExport2.xml')
            corpus = WordsToLetter(corpus)
            dataset = np.array(corpus)
        else: # Corpus en otomi
            base = 'corpus_otomi_'
            corpus = get_corpus(base + 'mod', corpora_path)
            hard_corpus = get_corpus(base + 'hard', corpora_path)
            corpus = WordsToLetter(corpus)
            hard_corpus = WordsToLetter(hard_corpus)
            dataset = np.array(corpus + hard_corpus, dtype=object)
        i = 0
        partial_time = 0
        accuracy_set = []
        kf = KFold(n_splits=hyper['k-folds'], shuffle=True)
        print("*"*10)
        print("K FOLDS VALIDATION")
        print("*"*10)
        for train_index, test_index in kf.split(dataset):
            i += 1
            print("\tK-Fold #", i)
            train_data, test_data = dataset[train_index], dataset[test_index]
            train_time, new_model_name = model_trainer(train_data, models_path,
                                                       hyper, verbose, i)
            y_test, y_pred, tagger = model_tester(test_data, models_path,
                                                  hyper, new_model_name,
                                                  verbose)
            accuracy_set.append(accuracy_score(y_test, y_pred))
            partial_time += train_time
            if verbose:
                print("*"*10)
                print("Partial Time>>", train_time, "Accuracy parcial>>",
                      accuracy_set[i - 1])
                eval_labeled_positions(y_test, y_pred)
                print(bio_classification_report(y_test, y_pred))
        print("Accuracy Set -->", accuracy_set)
        accuracy = sum(accuracy_set) / len(accuracy_set)
        train_time_format = str(round(partial_time / 60, 2)) + "[m]"
        train_size = len(train_data)
        test_size = len(test_data)
        print("Time>>", train_time_format, "Accuracy>>", accuracy)
        write_report(new_model_name, train_size, test_size, accuracy,
                     train_time_format, hyper)


if __name__ == '__main__':
    cli()

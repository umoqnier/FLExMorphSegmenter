# Diego Alberto Barriga Martinez @umoqnier
# Replica del experimento  'Automatic Prediction of Lezgi Morpheme Breaks'
# realizado para la lengua Lezgi para generar glosa automática a partir
# de escasos ejemplos

import os
import time
import pycrfsuite
import click
import json
from collections import Counter
from sklearn.model_selection import train_test_split
from utils import (get_hard_corpus, get_vic_data, WordsToLetter,
                   sent2features, sent2labels, extractTokens,
                   extractLabels, extractFeatures, eval_labeled_positions,
                   bio_classification_report, labels_decoder,
                   print_state_features, print_transitions, accuracy_score,
                   write_report)


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
@click.option('-d', '--debug', is_flag=True,
              help='Habilita el modo depuración')
@click.option('-v', '--verbose', is_flag=True,
              help='Habilita el modo verboso')
@click.argument('hparams', type=click.File('r'))
def cli(corpora_path, models_path, model_name, debug, verbose, test_size,
        iterations, l1, l2, hparams):
    """Command Line Interface para glosador automático del otomí (hñahñu)
    """
    if debug:
        breakpoint()
    hyper = json.loads(hparams.read())
    if model_name is None:
        model_name = hyper['name']

    vic_data = get_vic_data(corpora_path, hyper['dataset-train'])
    # Randomize and split the data
    if 'hard' in hyper['dataset-test']:
        train_data = WordsToLetter(vic_data)
        hard_corpus = get_hard_corpus(corpora_path, hyper['dataset-test'])
        test_data = WordsToLetter(hard_corpus, True)
        hyper['test-size'] = 'N/A'
    else:
        train_data, test_data = train_test_split(WordsToLetter(vic_data),
                                                 test_size=hyper['test-size'])

    X_train = sent2features(train_data)
    y_train = sent2labels(train_data)

    X_test = sent2features(test_data)
    y_test = sent2labels(test_data)

    # Train the model

    trainer = pycrfsuite.Trainer(verbose=verbose)

    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)

    # Set training parameters. L-BFGS is default. Using Elastic Net (L1 + L2)
    # regularization [ditto?].
    trainer.set_params({
            'c1': l1 or hyper['L1'],  # coefficient for L1 penalty
            'c2': l2 or hyper['L2'],  # coefficient for L2 penalty
            'max_iterations': iterations or hyper['max-iter']  # early stopping
        })
    compositive_name = f"tsu_{model_name}_{iterations or hyper['max-iter']}_{hyper['L1']}_{hyper['L2']}.crfsuite"
    # The program saves the trained model to a file:
    if not os.path.isfile(models_path + compositive_name):
        print(f"Entrenando nuevo modelo '{compositive_name}'")
        start = time.time()
        trainer.train(models_path + compositive_name)
        end = time.time()
        train_time = end - start
        train_time_format = str(round(train_time / 60, 2)) + "[m]"
        print("Fin de entrenamiento. Tiempo de entrenamiento >>", train_time,
              "[s]", train_time / 60, "[m]")
    else:
        train_time_format = "N/A"
        print("Usando modelo pre-entrenado >>", models_path + compositive_name)

    # ### Make Predictions
    tagger = pycrfsuite.Tagger()
    tagger.open(models_path + compositive_name)  # Passing model to tagger

    # First, let's use the trained model to make predications for just one
    # example sentence from the test data.
    # The predicted labels are printed out for comparison above the correct
    # labels. Most examples have 100% accuracy.

    if verbose:
        print("Basic example of prediction")
        example_sent = test_data[0]
        print('Letters:', '  '.join(extractTokens(example_sent)), end='\n')

        print('Predicted:',
              ' '.join(tagger.tag(extractFeatures(example_sent))), end='\n')
        print('Correct:', ' '.join(extractLabels(example_sent, 1)))

    # First, we will predict BIO labels in the test data:

    y_pred = []
    y_test = labels_decoder(y_test)
    for i, xseq in enumerate(X_test):
        y_pred.append(tagger.tag(xseq))

    accuracy = accuracy_score(y_test, y_pred)
    write_report(compositive_name, accuracy, train_time_format, hyper)
    # Get results for labeled position evaluation. This evaluates how well
    # the classifier performed on each morpheme as a whole and their tags,
    # rather than evaluating character-level. Then, we check the results and
    # print a report of the results. These results are for character level.
    if verbose:
        eval_labeled_positions(y_test, y_pred)

        print(bio_classification_report(y_test, y_pred))

        print("Accuracy Score>>>> ", accuracy_score(y_test, y_pred))

        info = tagger.info()

        # Prints top 15 labels transitions with learned transitions weights
        print("Top likely transitions:")
        print_transitions(Counter(info.transitions).most_common(15))

        print("\nTop unlikely transitions:")
        print_transitions(Counter(info.transitions).most_common()[-15:])

        print("Top positive:")
        print_state_features(Counter(info.state_features).most_common(15))

        print("\nTop negative:")
        print_state_features(Counter(info.state_features).most_common()[-15:])


if __name__ == '__main__':
    cli()

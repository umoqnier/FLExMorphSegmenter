# -*- coding: utf-8 -*-
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
@click.option('-d', '--debug', is_flag=True,
              help='Habilita el modo depuración')
@click.option('-i', '--iterations', type=int,
             help='Número máximo de iteraciones de entrenamiento')
@click.option('-v', '--verbose', is_flag=True,
              help='Habilita el modo verboso')
@click.argument('hparams', type=click.File('r'), default="hyperparams.json")
def cli(corpora_path, models_path, model_name, debug, verbose, iterations,
        hparams):
    """Command Line Interface para glosador automático del otomí (hñahñu)
    """
    hyper = json.loads(hparams.read())
    if model_name is None:
        model_name = hyper['name']

    # Randomize and split the data
    vic_data = get_vic_data(corpora_path, hyper['dataset-train'])
    train_data, test_data = train_test_split(WordsToLetter(vic_data, False),
                                             test_size=hyper['split-ratio'])
    hard_corpus = get_hard_corpus(corpora_path, hyper['dataset-test'])
    hard_data = WordsToLetter(hard_corpus, True)

    if debug:
        with open("train.txt", "w") as f:
            for t in train_data:
                f.write(str(t) + '\n')

        with open("test.txt", "w") as f:
            for t in test_data:
                f.write(str(t) + '\n')


    X_train = sent2features(train_data)
    y_train = sent2labels(train_data)

    X_test = sent2features(test_data)
    y_test = sent2labels(test_data)

    X_hard = sent2features(hard_data)
    y_hard = sent2labels(hard_data)

    # Train the model

    trainer = pycrfsuite.Trainer(verbose=debug)

    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)


    # Set training parameters. L-BFGS is default. Using Elastic Net (L1 + L2)
    # regularization [ditto?].
    trainer.set_params({
            'c1': hyper['L1'],  # coefficient for L1 penalty
            'c2': hyper['L2'],  # coefficient for L2 penalty
            'max_iterations': iterations or hyper['max-iter']  # early stopping
        })
    compositive_name = f"tsu_{model_name}_{iterations or hyper['max-iter']}_{hyper['L1']}_{hyper['L2']}.crfsuite"
    # The program saves the trained model to a file:
    if not os.path.isfile(models_path + compositive_name):
        print("ENTRENANDO...")
        start = time.time()
        trainer.train(models_path + compositive_name)
        end = time.time()
        train_time = end - start
        print("Fin de entrenamiento. Tiempo de entrenamiento >>", train_time,
              "[s]", train_time / 60, "[m]")
    else:
        print("Usando modelo pre-entrenado >>", models_path + compositive_name)

    # ### Make Predictions
    tagger = pycrfsuite.Tagger()
    tagger.open(models_path + compositive_name)  # Passing model to tagger

    # First, let's use the trained model to make predications for just one
    # example sentence from the test data.
    # The predicted labels are printed out for comparison above the correct
    # labels. Most examples have 100% accuracy.

    print("Basic example of prediction")
    example_sent = hard_data[0]
    print('Letters:', '  '.join(extractTokens(example_sent)), end='\n')

    print('Predicted:',
          ' '.join(tagger.tag(extractFeatures(example_sent))), end='\n')
    print('Correct:', ' '.join(extractLabels(example_sent, 1)))

    # First, we will predict BIO labels in the test data:

    y_pred_hard = []
    y_hard = labels_decoder(y_hard)
    for i, xseq in enumerate(X_hard):
        y_pred_hard.append(tagger.tag(xseq))

    accuracy = accuracy_score(y_hard, y_pred_hard)
    write_report(compositive_name, accuracy, train_time, hyper)
    # Get results for labeled position evaluation. This evaluates how well
    # the classifier performed on each morpheme as a whole and their tags,
    # rather than evaluating character-level. Then, we check the results and
    # print a report of the results. These results are for character level.
    if verbose:
        eval_labeled_positions(y_hard, y_pred_hard)

        print(bio_classification_report(y_hard, y_pred_hard))

        print("Accuracy Score>>>> ", accuracy_score(y_hard, y_pred_hard))

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

import os
import time
import pycrfsuite
import pickle
from collections import Counter
from itertools import chain
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import xml.etree.ElementTree as ET


def get_corpus(file_name, path='corpora/'):
    """ Obtiene el corpus con el nombre indicado

    Cada renglon del archivo es una oración codificada con glosa por cada
    fragmento de palabras y con su etiqueta POS por oración.
    El formato por renglon puede ser el siguiente:
        `[[[letras, glosa], [letras, glosa], ..., POS],...]`
    o
        `[[[[letras, glosa], [letras, glosa]], ..., POS],...]`
    :param path: Carpeta donde se encuentra el corpus
    :type: str
    :param file_name: Nombre del archivo que contiene el corpus
    :type: str
    :return: Lista donde cada elemento es un renglon el corpus
    :rtype: list
    """
    corpus_path = os.path.join(path, file_name)
    with open(corpus_path, encoding='utf-8', mode='r') as f:
        plain_text = f.read()
    if 'mod' in file_name:
        raw_data = plain_text.split('\n')
        return [eval(row) for row in raw_data if row]
    elif 'hard' in file_name:
        data = eval(plain_text)
        for frase in data:
            for i, chunk in enumerate(frase):
                pos_tag = chunk.pop(-1)
                chunk = chunk[0]
                chunk.append(pos_tag)
                frase[i] = chunk
        return data


def get_train_test(data, test_size, datatest, corpora_path):
    """
    Obtiene el conjunto de entrenamiento y de test

    Conjunto de entrenamiento y pruebas para la evaluación
    *hold-out*

    :param data: Corpus a evaluar
    :type: list
    :param test_size: Porcentaje del conjunto de pruebas
    :type: float
    :param datatest: Nombre del tipo de conjunto de pruebas que se utilizará
    :type: str
    :param corpora_path: Ruta de la carpeta que contiene los corpus
    :type: str
    :return: Tupla con dos listas. Una con el conjunto de entrenamiento
    y otra con el de pruebas
    :rtype: tuple
    """
    train_data, test_data = train_test_split(WordsToLetter(data),
                                             test_size=test_size)
    if 'hard' in datatest:
        hard_corpus = get_hard_corpus(corpora_path, datatest)
        # Add hard cases to test dataset
        hard_letters = WordsToLetter(hard_corpus, True)
        test_data = test_data + hard_letters
    return train_data, test_data


def param_setter(hyper, model_name, iterations, l1, l2):
    """ Setea parametros del CLI en diccionario

    Si existen parametros del CLI los setea en el diccionario hyper que es
    el que se encarga manejar los hiperparametros

    :param hyper: diccionario encargado de manejar los hiperparametros
    :type: dict
    :param model_name: nombre del modelo de entrenamiento
    :type: str
    :param iterations: Número de iteraciones maximas para entrenamiento
    :type: int
    :param l1: Parametro de penalización Elasticnet L1
    :type: float
    :param l2: Parámetro de penalización Elasticnet L2
    :type: float
    :return: Diccionario hyper modificado si existen parámetros en CLI
    :rtype: dict
    """
    if model_name:
        hyper['name'] = model_name
    if iterations:
        hyper['iterarions'] = iterations
    if l1:
        hyper['L1'] = l1
    if l2:
        hyper['L2'] = l2
    return hyper


def model_trainer(train_data, models_path, hyper, verbose, k):
    """ Entrena un modelo y lo guarda en disco

    Función encargada de entrenar un modelo con base en los hyperparametro y
    lo guarda como un archivo utilizable por `pycrfsuite`

    Parameters
    ----------
    train_data : list
    models_path : str
    hyper : dict
    verbose : bool
    k : int

    Returns
    -------
    train_time : float
        Tiempo de entrenamiento
    compositive_name : str
        Nombre del modelo entrenado
    """
    X_train = sent2features(train_data)
    y_train = sent2labels(train_data)

    # Train the model

    trainer = pycrfsuite.Trainer(verbose=verbose)

    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)

    # Set training parameters. L-BFGS is default. Using Elastic Net (L1 + L2)
    # regularization [ditto?].
    trainer.set_params({
            'c1': hyper['L1'],  # coefficient for L1 penalty
            'c2': hyper['L2'],  # coefficient for L2 penalty
            'max_iterations': hyper['max-iter']  # early stopping
        })
    # Setting model name
    model_name = f"{hyper['name']}_"
    if hyper['L1'] == 0 and hyper['L2'] == 0:
        model_name += f"noreg_"
    elif hyper['L1'] == 0:
        model_name += f"l1_zero_"
    elif hyper['L2'] == 0:
        model_name += f"l2_zero_"
    else:
        model_name += "regularized_"
    model_name += f"k_{k}.crfsuite"
    # The program saves the trained model to a file:
    path = models_path + hyper["name"] + f"/{model_name}"
    if not os.path.isfile(path):
        print(f"Entrenando nuevo modelo '{model_name}'")
        start = time.time()
        trainer.train(path)
        end = time.time()
        train_time = end - start
        print("Fin de entrenamiento. Tiempo de entrenamiento >>", train_time,
              "[s]", train_time / 60, "[m]")
    else:
        train_time = 0
        print("Usando modelo pre-entrenado >>", path)
    return train_time, model_name


def model_tester(test_data, models_path, hyper, model_name, verbose):
    """ Prueba un modelo preentrenado

    Recibe los datos de prueba y realiza las pruebas con el modelo previo

    Parameters
    ----------
    test_data : list
    models_path : str
    model_name : str
    verbose : bool

    Returns
    -------
    y_test : list
        Etiquetas reales
    y_pred : list
        Etiquetas predichas por el modelo
    tagger : Object
        Objeto que etiqueta con base en el modelo
    """
    X_test = sent2features(test_data)
    y_test = sent2labels(test_data)

    # ### Make Predictions
    tagger = pycrfsuite.Tagger()
    tag_path = os.path.join(models_path, hyper["name"], model_name)
    tagger.open(tag_path)  # Passing model to tagger

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
    for xseq in X_test:
        try:
            y_pred.append(tagger.tag(xseq))
        except UnicodeDecodeError as e:
            print("ERROR al producir etiquetas")
            print(e.object)
            print(e.reason)

    return y_test, y_pred, tagger


def report_printer(y_test, y_pred, tagger):
    """
    """
    # Get results for labeled position evaluation. This evaluates how well
    # the classifier performed on each morpheme as a whole and their tags,
    # rather than evaluating character-level. Then, we check the results and
    # print a report of the results. These results are for character level.
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


def XMLtoWords(filename):
    """
    Takes FLExText text as .xml. Returns data as list: [[[[[[morpheme, gloss], pos],...],words], sents]].
    Ignores punctuation. Morph_types can be: stem, suffix, prefix, or phrase when lexical item is made up of two words.
    """

    datalists = []

    # open XML doc using xml parser
    root = ET.parse('legacy/' + filename).getroot()

    for text in root:
        for paragraphs in text:
            # Only get paragraphs, ignore metadata.
            if paragraphs.tag == 'paragraphs':
                for paragraph in paragraphs:
                    # jump straight into items under phrases
                    for j, phrase in enumerate(paragraph[0]):
                        sent = []
                        # ignore first item tag which is the sentence number
                        for i, word in enumerate(phrase[1]):
                            # ignore punctuation tags which have no attributes
                            if word.attrib:
                                lexeme = []
                                for node in word:
                                    if node.tag == 'morphemes':
                                        for morph in node:
                                            morpheme = []
                                            # note morph type
                                            morph_type = morph.get('type')
                                            # Treat MWEs or unlabled morphemes as stems.
                                            if morph_type == None or morph_type == 'phrase':
                                                morph_type = 'stem'
                                            for item in morph:
                                                # get morpheme token
                                                if item.get('type') == 'txt':
                                                    form = item.text
                                                    # get rid of hyphens demarcating affixes
                                                    if morph_type == 'suffix':
                                                        form = form[1:]
                                                    if morph_type == 'prefix':
                                                        form = form[:-1]
                                                    morpheme.append(form)
                                                # get affix glosses
                                                if item.get('type') == 'gls' and morph_type != 'stem':
                                                    morpheme.append(item.text)
                                            # get stem "gloss" = 'stem'
                                            if morph_type == 'stem':
                                                morpheme.append(morph_type)
                                            lexeme.append(morpheme)
                                    # get word's POS
                                    if node.get('type') == 'pos':
                                        lexeme.append(node.text)
                                sent.append(lexeme)
                        datalists.append(sent)
    return datalists


def WordsToLetter(wordlists):
    '''
    Takes data from XMLtoWords:
        `[[[[[[morpheme, gloss], pos],...],words],sents]]`
    Returns [[[[[letter, POS, BIO-label],...],words],sents]]
    '''

    letterlists = []
    for i, phrase in enumerate(wordlists):
        sent = []
        for lexeme in phrase:
            palabra = ''
            word = []
            #Skip POS label
            for morpheme in lexeme[:-1]:
                palabra += ''.join([l for l in morpheme[0]])
                #use gloss as BIO label
                label = morpheme[1]
                #Break morphemes into letters
                for i in range(len(morpheme[0])):
                    letter = [morpheme[0][i]]  # Adding assci for encoding
                    #add POS label to each letter
                    letter.append(lexeme[-1])
                    #add BIO label
                    if i == 0:
                        letter.append('B-' + label)
                    else:
                        letter.append('I-' + label)
                        #letter.append('I')
                    word.append(letter)
            sent.append(word)
        letterlists.append(sent)
    return letterlists


def extractFeatures(sent):
    ''' Reglas que configuran las feature functions para entrenamiento

    :param sent: Data as `[[[[[letter, POS, BIO-label],...],words],sents]]`
    :type: list
    :return: list of words with characters as features list:
        [[[[[letterfeatures],POS,BIO-label],letters],words]]
    :rtype: list
    '''

    featurelist = []
    senlen = len(sent)
    # each word in a sentence
    for i in range(senlen):
        word = sent[i]
        wordlen = len(word)
        lettersequence = ''
        # each letter in a word
        for j in range(wordlen):
            letter = word[j][0]
            # gathering previous letters
            lettersequence += letter
            # ignore digits
            if not letter.isdigit():
                features = [
                    'bias',
                    'letterLowercase=' + letter.lower(),
                ]
                # Position of word in sentence
                if i == senlen -1:
                    features.append("EOS")
                else:
                    features.append("BOS")

                # Pos tag sequence (Don't get pos tag if sentence is 1 word long)
                if i > 0 and senlen > 1:
                    features.append('prevpostag=' + sent[i-1][0][1])
                    if i != senlen-1:
                        features.append('nxtpostag=' + sent[i+1][0][1])

                # Position of letter in word
                if j == 0:
                    features.append('BOW')
                elif j == wordlen-1:
                    features.append('EOW')
                else:
                    features.append('letterposition=-%s' % str(wordlen-1-j))

                # Letter sequences before letter
                if j >= 4:
                    features.append('prev4letters=' + lettersequence[j-4:j].lower() + '>')
                if j >= 3:
                    features.append('prev3letters=' + lettersequence[j-3:j].lower() + '>')
                if j >= 2:
                    features.append('prev2letters=' + lettersequence[j-2:j].lower() + '>')
                if j >= 1:
                    features.append('prevletter=' + lettersequence[j-1:j].lower() + '>')

                # letter sequences after letter
                if j <= wordlen-2:
                    nxtlets = word[j+1][0]
                    features.append('nxtletter=<' + nxtlets.lower())
                if j <= wordlen-3:
                    nxtlets += word[j+2][0]
                    features.append('nxt2letters=<' + nxtlets.lower())
                if j <= wordlen-4:
                    nxtlets += word[j+3][0]
                    features.append('nxt3letters=<' + nxtlets.lower())
                if j <= wordlen-5:
                    nxtlets += word[j+4][0]
                    features.append('nxt4letters=<' + nxtlets.lower())

            # Add encoding for pysrfsuite
            featurelist.append([f.encode('utf-8') for f in features])
    return featurelist


def extractLabels(sent, flag=0):
    labels = []
    for word in sent:
        for letter in word:
            if flag:
                labels.append(letter[2])
            else:
                labels.append(letter[2].encode('utf-8'))
    return labels


def extractTokens(sent):
    tokens = []
    for word in sent:
        for letter in word:
            tokens.append(letter[0])
    return tokens


def sent2features(data):
    return [extractFeatures(sent) for sent in data]


def sent2labels(data):
    return [extractLabels(sent) for sent in data]


def sent2tokens(data):
    return [extractTokens(sent) for sent in data]


def bio_classification_report(y_correct, y_pred):
    '''Takes list of correct and predicted labels from tagger.tag.
    Prints a classification report for a list of BIO-encoded sequences.
    It computes letter-level metrics.'''

    labeler = LabelBinarizer()
    y_correct_combined = labeler.fit_transform(list(chain.from_iterable(y_correct)))
    y_pred_combined = labeler.transform(list(chain.from_iterable(y_pred)))

    tagset = set(labeler.classes_)
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(labeler.classes_)}

    return classification_report(
        y_correct_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset)


def concatenateLabels(y_list):
    '''Return list of morpheme
    :param y_list:
    :type: list
    :return: labels = [B-label,...]]
             morph = [[B-label, I-label, ...], [B-label, I-label, ...]]
    '''

    morphs_list = []
    labels_list = []
    morph = []
    for sent in y_list:
        for label in sent:
            labels_list.append(label)
            if label[0] == 'I':
                # build morpheme shape, adding to first letter
                morph.append(label)
            else:
                # Once processed first morph, add new morphemes & gloss labels to output
                if morph:
                    morphs_list.append(morph)
                # Extract morpheme features
                morph = [label]

    return morphs_list, labels_list


def countMorphemes(morphlist):
    """ Cuenta el número de ocurrencias de cada label

    :param morphlist: Lista de bio-labels
    :return: Diccionario con las labesl como llave y el número de
    ocurrencias como valor
    """
    counts = {}
    for morpheme in morphlist:
        label = morpheme[0][2:]
        counts[label] = counts.get(label, 0) + 1
    return counts


def eval_labeled_positions(y_correct, y_pred):
    """Imprime un reporte de metricas entre las bio labels predichas
    por el modelo y las reales

    Genera diccionarios con las labels como llaves y el total de ocurrencias
    como valores. Con estos diccionarios obtiene precision, recall y f-score
    por cada label. Además, obtiene dichas métricas para todas las labels.

    :param y_correct: Bio labels reales
    :type: list
    :param y_pred: Bio labels predichas por el modelo
    :type: list
    :return: None
    """
    # group the labels by morpheme and get list of morphemes
    correctmorphs, _ = concatenateLabels(y_correct)
    predmorphs, predLabels = concatenateLabels(y_pred)
    # Count instances of each morpheme
    test_morphcts = countMorphemes(correctmorphs)
    pred_morphcts = countMorphemes(predmorphs)

    correctMorphemects = {}
    idx = 0
    num_correct = 0
    for morpheme in correctmorphs:  # TODO: Improve this section
        correct = True
        for label in morpheme:
            if label != predLabels[idx]:
                correct = False
            idx += 1
        if correct == True:
            num_correct += 1
            correctMorphemects[morpheme[0][2:]] = correctMorphemects.get(morpheme[0][2:], 0) + 1
    # calculate P, R F1 for each morpheme
    results = ''
    for firstlabel in correctMorphemects.keys():
        # Calculate precision for each label
        lprec = correctMorphemects[firstlabel] / pred_morphcts[firstlabel]
        # Calculate Recall for each label
        lrecall = correctMorphemects[firstlabel] / test_morphcts[firstlabel]
        results += firstlabel + '\t\t{0:.2f}'.format(lprec) + '\t\t' + '{0:.2f}'.format(
            lrecall) + '\t' + '{0:.2f}'.format((2 * lprec * lrecall) / (lprec + lrecall)) + '\t\t' + str(
            test_morphcts[firstlabel]) + '\n'
    # overall results
    precision = num_correct / len(predmorphs)
    recall = num_correct / len(correctmorphs)

    print('\t\tPrecision\tRecall\tf1-score\tInstances\n\n' + results + '\ntotal/avg\t{0:.2f}'.format(
        precision) + '\t\t' + '{0:.2f}'.format(recall) + '\t' + '{0:.2f}'.format(
        (2 * precision * recall) / (precision + recall)))


def print_transitions(trans_features):
    '''Print info from the crfsuite.'''
    print("From Label -> To label | Weight")
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-8s | %0.6f" % (label_from, label_to, weight))


def print_state_features(state_features):
    print("Weight | Label | Attribute")
    for (attr, label), weight in state_features:
        print("%0.6f | %-6s | %s" % (weight, label, attr))


def sents_encoder(sent):
    inter = []
    result = []
    for seq in sent:
        for sub_seq in seq:
            aux = [item.encode("utf-8") for item in sub_seq]
            inter.append(aux)
        result.append(inter)
        inter = []
    return result


def sents_decoder(sent):
    inter = []
    result = []
    for seq in sent:
        for sub_seq in seq:
            aux = [item.decode("utf-8") for item in sub_seq]
            inter.append(aux)
        result.append(inter)
        inter = []
    return result


def labels_decoder(test):
    result = []
    for labels in test:
        aux = [label.decode("utf-8") for label in labels]
        result.append(aux)
    return result


def accuracy_score(y_test, y_pred):
    right, wrong, total = 0, 0, 0
    for tests, predictions in zip(y_test, y_pred):
        total += len(tests)
        for t, p in zip(tests, predictions):
            if t == p:
                right += 1
            elif t != p:
                wrong += 1

    return right / total


def write_report(model_name, train_size, test_size, accuracy, train_time,
                 hyper):
    """Escribe el reporte con resultados e hiperparametros

    """
    # Header
    line = ''
    line += model_name + ','
    line += str(round(accuracy, 4)) + ','
    line += train_time + ','
    line += str(hyper['L1']) + ','
    line += str(hyper['L2']) + ','
    line += hyper['dataset'] + ','
    line += str(train_size) + ','
    line += str(test_size) + ','
    line += str(hyper['max-iter']) + ','
    line += str(hyper['k-folds']) + ','
    line += hyper['description'] + "\n"
    with open('results.csv', 'a') as f:
        f.write(line)

from itertools import chain
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
import xml.etree.ElementTree as ET


def get_vic_data(path, file_name):
    with open(path + file_name, encoding='utf-8', mode='r') as f:
        plain_text = f.read()
    raw_data = plain_text.split('\n')
    return [eval(row) for row in raw_data]


def get_hard_corpus(path, file_name):
    with open(path + file_name, encoding='utf-8', mode='r') as f:
        plain_text = f.read()
    return eval(plain_text)


def corpus_verifier(corpus_list):
    pass


def XMLtoWords(filename):
    """
    Takes FLExText text as .xml. Returns data as list: [[[[[[morpheme, gloss], pos],...],words], sents]].
    Ignores punctuation. Morph_types can be: stem, suffix, prefix, or phrase when lexical item is made up of two words.
    """

    datalists = []

    #open XML doc using xml parser
    root = ET.parse(filename).getroot()

    for text in root:
        for paragraphs in text:
            #Only get paragraphs, ignore metadata.
            if paragraphs.tag == 'paragraphs':
                for paragraph in paragraphs:
                    #jump straight into items under phrases
                    for j, phrase in enumerate(paragraph[0]):
                        sent = []
                        # ignore first item tag which is the sentence number
                        for i, word in enumerate(phrase[1]):
                            #ignore punctuation tags which have no attributes
                            if word.attrib:
                                lexeme = []
                                for node in word:
                                    if node.tag == 'morphemes':
                                        for morph in node:
                                            morpheme = []
                                            #note morph type
                                            morph_type = morph.get('type')
                                            #Treat MWEs or unlabled morphemes as stems.
                                            if morph_type == None or morph_type == 'phrase':
                                                morph_type = 'stem'
                                            for item in morph:
                                                #get morpheme token
                                                if item.get('type') == 'txt':
                                                    form = item.text
                                                    #get rid of hyphens demarcating affixes
                                                    if morph_type == 'suffix':
                                                        form = form[1:]
                                                    if morph_type == 'prefix':
                                                        form = form[:-1]
                                                    morpheme.append(form)
                                                #get affix glosses
                                                if item.get('type') == 'gls' and morph_type != 'stem':
                                                    morpheme.append(item.text)
                                            #get stem "gloss" = 'stem'
                                            if morph_type == 'stem':
                                                morpheme.append(morph_type)
                                            lexeme.append(morpheme)
                                    #get word's POS
                                    if node.get('type') == 'pos':
                                        lexeme.append(node.text)
                                sent.append(lexeme)
                        datalists.append(sent)
    return datalists


def WordsToLetter(wordlists, flag=False):
    '''
    Takes data from XMLtoWords: [[[[[[morpheme, gloss], pos],...],words],sents]].
    Returns [[[[[letter, POS, BIO-label],...],words],sents]]
    '''

    #print("=================Words to letter")
    letterlists = []
    #print("Objeto recibido", wordlists[:3], "...")
    for i, phrase in enumerate(wordlists):
        #print(i, "phrase=================")
        sent = []
        for lexeme in phrase:
            if flag:
                pos = lexeme[-1]
                lexeme = lexeme.pop(0)
                lexeme.append(pos)
            palabra = ''
            word = []
            #Skip POS label
            for morpheme in lexeme[:-1]:
                palabra += ''.join([l for l in morpheme[0]])
                #use gloss as BIO label
                label = morpheme[1]
                #print("\t\tLabel >>", label)
                #Break morphemes into letters
                #print("\t\t*** LETTERS")
                for i in range(len(morpheme[0])):
                    letter = [morpheme[0][i]]  # Adding assci for encoding
                    #print("\t\t\tL >>", letter[0])
                    #add POS label to each letter
                    letter.append(lexeme[-1])
                    #print("\t\t\tAdd POS >>", letter)
                    #add BIO label
                    if i == 0:
                        letter.append('B-' + label)
                    else:
                        letter.append('I-' + label)
                        #letter.append('I')
                    #print("\t\tAdd BIO label >>", letter)
                    word.append(letter)
                    #print("\t*** WORD >>", word)
            sent.append(word)
            #print("*** SENT>>", sent)
        letterlists.append(sent)
    #for j, l in enumerate(letterlists[:5]):
        #print("sent #", j)
        #for i in l:
            #print(i)
    return letterlists


def extractFeatures(sent):
    ''' Reglas que configuran las feature functions para entrenamiento

    :param sent: Data as [[[[[letter, POS, BIO-label],...],words],sents]]
    :type: list
    :return: list of words with characters as features list:
        [[[[[letterfeatures],POS,BIO-label],letters],words]]
    :rtype: list
    '''

    featurelist = []
    senlen = len(sent)

    # TODO: Optimizar los parametros hardcode para el otomí.
    #each word in a sentence
    for i in range(senlen):
        word = sent[i]
        wordlen = len(word)
        lettersequence = ''
        #each letter in a word
        for j in range(wordlen):
            letter = word[j][0]
            #gathering previous letters
            lettersequence += letter
            #ignore     digits
            if not letter.isdigit():
                features = [
                    'bias',
                    'letterLowercase=' + letter.lower(),
                    'postag=' + word[j][1],
                ]
                #position of word in sentence and pos tags sequence
                if i > 0:
                    features.append('prevpostag=' + sent[i-1][0][1])
                    if i != senlen-1:
                        features.append('nxtpostag=' + sent[i+1][0][1])
                    else:
                        features.append('EOS')
                else:
                    features.append('BOS')
                    #Don't get pos tag if sentence is 1 word long
                    if i != senlen-1:
                        features.append('nxtpostag=' + sent[i+1][0][1])
                #position of letter in word
                if j == 0:
                    features.append('BOW')
                elif j == wordlen-1:
                    features.append('EOW')
                else:
                    features.append('letterposition=-%s' % str(wordlen-1-j))
                #letter sequences before letter
                if j >= 4:
                    features.append('prev4letters=' + lettersequence[j-4:j].lower() + '>')
                if j >= 3:
                    features.append('prev3letters=' + lettersequence[j-3:j].lower() + '>')
                if j >= 2:
                    features.append('prev2letters=' + lettersequence[j-2:j].lower() + '>')
                if j >= 1:
                    features.append('prevletter=' + lettersequence[j-1:j].lower() + '>')
                #letter sequences after letter
                if j <= wordlen-2:
                    nxtlets = word[j+1][0]
                    features.append('nxtletter=<' + nxtlets.lower())
                    #print('\nnextletter:', nxtlet)
                if j <= wordlen-3:
                    nxtlets += word[j+2][0]
                    features.append('nxt2letters=<' + nxtlets.lower())
                    #print('next2let:', nxt2let)
                if j <= wordlen-4:
                    nxtlets += word[j+3][0]
                    features.append('nxt3letters=<' + nxtlets.lower())
                if j <= wordlen-5:
                    nxtlets += word[j+4][0]
                    features.append('nxt4letters=<' + nxtlets.lower())
            featurelist.append([f.encode('utf-8') for f in features])  # Add encoding for pysrfsuite
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
    right, wrong, total, control = 0, 0, 0, 0
    for tests, predictions in zip(y_test, y_pred):
        if len(tests) == len(predictions):
            control += 1
        total += len(tests)
        for t, p in zip(tests, predictions):
            if t == p:
                right += 1
            elif t != p:
                wrong += 1

    return right / total


def write_report(model_name, accuracy, train_time, hyper):
    """Escribe el reporte con resultados e hiperparametros

    """
    line = model_name + "," + hyper['dataset-train'] + "," + \
            hyper['dataset-test'] + "," + str(train_time / 60) + "," + \
            str(hyper['max-iter']) + "," + str(hyper['L1']) + "," + \
            str(hyper['L2']) + "," + str(accuracy) + "," + \
            hyper['description'] + "\n"
    with open('results.csv', 'a') as f:
        f.write(line)

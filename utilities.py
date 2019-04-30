from itertools import chain
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

def get_vic_data():
    with open("corpusOtomi.txt") as f:
        plain_text = f.read()
    raw_data = plain_text.split('\n')
    return [eval(row) for row in raw_data]



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
                    #print(paragraph.tag, paragraph.attrib)
                    for j, phrase in enumerate(paragraph[0]):
                        #print(j, '>>', phrase.tag, phrase.attrib)
                        sent = []
                        # ignore first item tag which is the sentence number
                        for i, word in enumerate(phrase[1]):
                           # print('\t', i, ':', word.tag, word.attrib)
                            #ignore punctuation tags which have no attributes
                            if word.attrib:
                                lexeme = []
                                for node in word:
                                    if node.tag == 'morphemes':
                                        #print("\t\tCASO MORFEMA")
                                        for morph in node:
                                            morpheme = []
                                            #note morph type 
                                            morph_type = morph.get('type')
                                            #print("\t\tTipo >>", morph_type)
                                            #Treat MWEs or unlabled morphemes as stems.
                                            if morph_type == None or morph_type == 'phrase':
                                                morph_type = 'stem'
                                            for item in morph:
                                                #get morpheme token
                                                if item.get('type') == 'txt':
                                                   # print("\t\t\tTXT >>", item.text)
                                                    form = item.text
                                                    #get rid of hyphens demarcating affixes
                                                    if morph_type == 'suffix':
                                                        form = form[1:]
                                                       # print("\t\t\tSUFFIX >> ", form)
                                                    if morph_type == 'prefix':
                                                        form = form[:-1]
                                                       # print("\t\t\tPREFIX >> ", form)
                                                    morpheme.append(form)
                                                #get affix glosses
                                                if item.get('type') == 'gls' and morph_type != 'stem':
                                                    morpheme.append(item.text)
                                            #get stem "gloss" = 'stem'
                                            if morph_type == 'stem':
                                                morpheme.append(morph_type)
                                            lexeme.append(morpheme)
                                           # print("*** Current LEXEME", lexeme)
                                       # print("\t\t>>> FINISH MORFEMA")
                                    #get word's POS
                                    if node.get('type') == 'pos':
                                       # print("\t\tCASO POS TAG")
                                        lexeme.append(node.text)
                                       # print("*** Current LEXEME", lexeme)
                                    #print("\t\t>>> FINISH NODE ", node.tag)
                                sent.append(lexeme)
                                #print("\t>>>Finish WORD")
                               # print('**** CURRENT SENT', sent)
                        datalists.append(sent)
    # print(datalists[:10])
    return datalists


def WordsToLetter(wordlists):
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
            word = []
            #Skip POS label
            for morpheme in lexeme[:-1]:
                #print("\tMorfema >>", morpheme)
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
    '''Takes data as [[[[[letter, POS, BIO-label],...],words],sents]].
    Returns list of words with characters as features list: [[[[[letterfeatures],POS,BIO-label],letters],words]]'''
    
    featurelist = []
    senlen = len(sent)

    # TODO: Optimizar los parametros hardcode para el otomí. Se probaran nuevos parámetro
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
            #ignore digits             
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
    '''Return list of morpheme labels [[B-label, I-label,...]morph,[B-label,...]]'''

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
    counts = {}
    for morpheme in morphlist:
        counts[morpheme[0][2:]] = counts.get(morpheme[0][2:], 0) + 1
    return counts


def eval_labeled_positions(y_correct, y_pred):
    # group the labels by morpheme and get list of morphemes
    correctmorphs, _ = concatenateLabels(y_correct)
    predmorphs, predLabels = concatenateLabels(y_pred)
    # Count instances of each morpheme
    test_morphcts = countMorphemes(correctmorphs)
    pred_morphcts = countMorphemes(predmorphs)

    correctMorphemects = {}
    idx = 0
    num_correct = 0
    for morpheme in correctmorphs:
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
        lprec = correctMorphemects[firstlabel] / pred_morphcts[firstlabel]
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
    
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))


def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-6s %s" % (weight, label, attr))


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


def test_decoder(test):
    result = []
    for labels in test:
        aux = [label.decode("utf-8") for label in labels]
        result.append(aux)
    return result

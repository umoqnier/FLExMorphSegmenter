
# coding: utf-8

# # Automatic Prediction of Lezgi Morpheme Breaks
# 
# This program does supervised morphological analysis and glossing of affixes. It is intended to quickly increase the amount of accessible data from low resource, and often endangered, languages. This classifier can be used on any language but it expects 2000-3000 words of cleanly annotated data. 
# 
# This example is designed for Lezgi [lez], a Nakh-Daghestanian language spoken in Russia and Azerbaijan. Lezgi is an agglutinating language that is overwhelmingly suffixing. The training and test data came from a collection of 21 transcribed oral narratives spoken in the Qusar dialect of northwest Azerbaijan. Nine texts with about 2,500 words were used for training data after having been cleanly annotated with morpheme breaks and part of speech. All but three of affixes were glossed. Many of the stems are not glossed. The FlexText XML export labels each morpheme as stem, suffix, or prefix. 
# 
# This program is considered successful if it reaches 80% accuracy. This goal comes from the Pareto Principle - the idea that 20% of one's effort produces 80% of one's results, and vice versa. This program should accurately complete 80% of the annotations, leaving the most interesting and informative 20% for the human linguist to complete.This project was inspired by an ongoing fieldwork project. A native Lezgi speaker who has no background in linguistics has been annotating the collection of texts. She has quickly learned basic morphology and gained FLEx skills. However, simultaneously learning and doing basic linguistic analysis produces inaccurate and inconsistent annotations. It is also time-consuming. Many of the mistakes are due to the repetitive nature of the work. Not every part of speech has inflectional morphology. The annotator is most likely to skip over essential words with simple morphology, such as ergative case-marked arguments, and concentrate on morphologicaly complex words. 
# 
# Once the training is complete, the program should predict morpheme breaks and affix glosses for any text that has been labeled with parts of speech. Identifying parts of speech is required because this seems a reasonable task for a non-linguist native speaker. The data used in this example does include two distinctions in Lezgi that might be difficult without linguistic training. Participles are distinguished from verbs, but Lezgi participles end in a unique letter. Demonstrative pronouns are distinguished from pronouns. This distinction was used primarily because it was already consistently annotated in the data. 
# 

#  ## Preprocessing Data
#  
# This process assumes that 1) the data has been analyzed in FLEx and exported as a FlexText, then saved with an .xml file extension, 2) words have been annotated in FLEx for part of speech, (for this example - verb, participle, adjective, adverb, noun/proper noun, particle, (personal) pronoun, demonstrative, and postposition), 3) morpheme breaks are consistent, and 4) all affixes, but not stems, are glossed.

# In[1]:

#API for parsing XML docs
import xml.etree.ElementTree as ET
from itertools import chain
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite
from collections import Counter


# In[2]:

def XMLtoWords(filename):
    '''Takes FLExText text as .xml. Returns data as list: [[[[[[morpheme, gloss], pos],...],words],sents]].
    Ignores punctuation. Morph_types can be: stem, suffix, prefix, or phrase when lexical item is made up of two words.'''
    
    datalists = []

    #open XML doc using xml parser
    root = ET.parse(filename).getroot()

    for text in root:
        for paragraphs in text:
            #Only get paragraphs, ignore metadata.
            if paragraphs.tag == 'paragraphs':
                for paragraph in paragraphs:
                    #jump straight into items under phrases
                    for phrase in paragraph[0]:
                        sent = []
                        #ignore first item which is the sentence number
                        for word in phrase[1]:
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


# In[3]:

def WordsToLetter(wordlists):
    '''Takes data from XMLtoWords: [[[[[[morpheme, gloss], pos],...],words],sents]]. 
    Returns [[[[[letter, POS, BIO-label],...],words],sents]]'''

    letterlists = []
    
    for phrase in wordlists:
        sent = []
        for lexeme in phrase:
            word = []
            #Skip POS label
            for morpheme in lexeme[:-1]:
                #use gloss as BIO label
                label = morpheme[1]
                #Break morphemes into letters
                for i in range(len(morpheme[0])):
                    letter = [morpheme[0][i]]
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


# The call below takes the data from the FLExText XML export. The data is read from the XML file and broken down by morphemes. Then it is broken down by letter. Each letter is associated with the word's part of speech tag and a BIO label. The BIO label for stems is "stem". The label for affixes is their gloss. "B" denotes the initial letter of a morpheme. I marks non-initial letters.
# 
# With a corpus of a little less than 2,500 words, I originally tried a 90/10 split. The accuracy results ranged from 92% to 97% but the test data was seeing a dozen or less labels. An 80/20 random split ranges less than 2% in accuracy, but still averages about 94%. However, the number of labels the test data encounters is nearly doubled.

# In[4]:

#Randomize and split the data
traindata,testdata = train_test_split(WordsToLetter(XMLtoWords("FLExTxtExport2.xml")),test_size=0.2)


# ## CRFSuite 
# ### Define Features
# 
# It is assumed that a "phrase" in FLEx is equivalent to a complete sentence. In reality, some "phrases" contain more than one sentence, some contain only a sentence fragment. This means that the word position in the sentence is often inaccurate, but it was retained to take into account Lezgi's strong tendency for verb-final word order. Affixes are rarely more than 3 letters long, so features include the previous and next 1-4 letters. This ensures that the program is viewing at least one letter in the previous/next morpheme. More often it is viewing the whole previous/next 1-2 morphemes. 
# 
# Since Lezgi is primarily suffixing, the position of a letter in a word is counted from the end of the word. 

# In[5]:

def extractFeatures(sent):
    '''Takes data as [[[[[letter, POS, BIO-label],...],words],sents]].
    Returns list of words with characters as features list: [[[[[letterfeatures],POS,BIO-label],letters],words]]'''
    
    featurelist = []
    senlen = len(sent)
    
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
                
            featurelist.append(features)
    
    return featurelist

def extractLabels(sent):
    labels = []
    for word in sent:
        for letter in word:
            labels.append(letter[2])
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


# In[6]:

X_train = sent2features(traindata)
Y_train = sent2labels(traindata)

X_test = sent2features(testdata)
Y_test = sent2labels(testdata)


# ### Train the model

# In[7]:

trainer = pycrfsuite.Trainer(verbose=False)

for xseq, yseq in zip(X_train, Y_train):
    trainer.append(xseq, yseq)


# Set training parameters. L-BFGS (what is this) is default. Using Elastic Net (L1 + L2) regularization [ditto?].

# In[8]:

trainer.set_params({
        'c1': 1.0, #coefficient for L1 penalty
        'c2': 1e-3, #coefficient for L2 penalty
        'max_iterations': 50 #early stopping
    })


# The program saves the trained model to a file:

# In[9]:

model_filename = 'LING5800_lezgi.crfsuite'
trainer.train(model_filename)


# ### Make Predictions

# In[10]:

tagger = pycrfsuite.Tagger()
tagger.open(model_filename)


# First, let's use the trained model to make predications for just one example sentence from the test data. The predicted labels are printed out for comparison above the correct labels. Most examples have 100% accuracy.

# In[11]:

example_sent = testdata[0]
print('Letters:', '  '.join(extractTokens(example_sent)), end='\n')

print('Predicted:', ' '.join(tagger.tag(extractFeatures(example_sent))))
print('Correct:', ' '.join(extractLabels(example_sent)))


# ## Evaluate the Model
# 
# The following function will evaluate how well the model performs. Unlike CRF example found at https://github.com/scrapinghub/python-crfsuite/blob/master/examples/CoNLL%202002.ipynb, this model is not designed to disregard "O" labels, since all characters that are not part of a word (e.g. digits and punctuation) are already eliminated during pre-processing.

# In[12]:

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


# First, we will predict BIO labels in the test data:

# In[13]:

Y_pred = [tagger.tag(xseq) for xseq in X_test]


# Get results for labeled position evaluation. This evaluates how well the classifier performed on each morpheme as a whole and their tags, rather than evaluating character-level.

# In[14]:

def concatenateLabels(y_list):
    '''Return list of morpheme labels [[B-label, I-label,...]morph,[B-label,...]]'''
    
    morphs_list = []
    labels_list = []
    morph = []
    for sent in y_list:
        for label in sent:
            labels_list.append(label)
            if label[0] == 'I':
                #build morpheme shape, adding to first letter
                morph.append(label)
            else:
                # Once processed first morph, add new morphemes & gloss labels to output
                if morph:
                    morphs_list.append(morph)
                #Extract morpheme features
                morph = [label]
    
    return morphs_list, labels_list

def countMorphemes(morphlist):
    counts = {}
    for morpheme in morphlist:
        counts[morpheme[0][2:]] = counts.get(morpheme[0][2:], 0) + 1
    return counts

def eval_labeled_positions(y_correct, y_pred):
    
    #group the labels by morpheme and get list of morphemes
    correctmorphs,_ = concatenateLabels(y_correct)
    predmorphs,predLabels = concatenateLabels(y_pred)
    #Count instances of each morpheme
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
    #calculate P, R F1 for each morpheme
    results = ''
    for firstlabel in correctMorphemects.keys():
        lprec = correctMorphemects[firstlabel]/pred_morphcts[firstlabel]
        lrecall = correctMorphemects[firstlabel]/test_morphcts[firstlabel]
        results += firstlabel + '\t\t{0:.2f}'.format(lprec) + '\t\t' + '{0:.2f}'.format(lrecall) + '\t' + '{0:.2f}'.format((2*lprec*lrecall)/(lprec+lrecall)) +'\t\t' + str(test_morphcts[firstlabel]) + '\n'
    #overall results
    precision = num_correct/len(predmorphs)
    recall = num_correct/len(correctmorphs)
    
    print('\t\tPrecision\tRecall\tf1-score\tInstances\n\n' + results + '\ntotal/avg\t{0:.2f}'.format(precision) + '\t\t' + '{0:.2f}'.format(recall) + '\t' + '{0:.2f}'.format((2*precision*recall)/(precision+recall)))


# Then, we check the results and print a report of the results. These results are for character level.

# In[15]:

eval_labeled_positions(Y_test, Y_pred)


# In[173]:

print(bio_classification_report(Y_test, Y_pred))


# The model, with a 80/20 split, produces an average accuracy of 94% with a less than 2% range over randomized test data. This is significantly above the targeted accuracy of 80%. Table 1 shows the results of one run. 
# 
# |__label__|__precision__|__recall__|__f1-score__|__instances__|
# |---------|-------------|----------|------------|-------------|
# |B-AOR|1.00|0.88|0.94|17|
# |B-DAT|0.92|1.00|0.96|11|
# |B-ELAT|0.67|1.00|0.80|2|
# |B-ENT|0.33|0.50|0.40|2|
# |B-ERG|0.00|0.00|0.00|3|
# |B-FOC|0.86|1.00|0.92|6|
# |B-FUT|0.00|0.00|0.00|2|
# |B-GEN|0.50|0.33|0.40|6|
# |B-HORT|0.00|0.00|0.00|1|
# |I|0.95|0.99|0.97|480|
# |B-INESS|1.00|0.33|0.50|3|
# |B-MSDR|0.00|0.00|0.00|2|
# |B-NEG|1.00|1.00|1.00|1|
# |B-OBL|0.80|0.60|0.69|20|
# |B-PL|0.50|0.50|0.50|2|
# |B-POESS|1.00|1.00|1.00|3|
# |B-PTP|1.00|0.67|0.80|3|
# |B-SBST|1.00|0.50|0.67|2|
# |B-SUPER|0.67|1.00|0.80|2|
# |B-TEMP|0.00|0.00|0.00|1|
# |B-UNK|0.00|0.00|0.00|1|
# |B-stem|1.00|0.99|0.99|138|
# |__avg / total__|__0.94__|__0.94__|__0.94__|__708__|
# 
# <center>Table 1: Results of morpheme predictions</center>
# 
# As might be expected, the classifier has less success predicting less frequent labels. This makes the results of the I labels (non-initial letters in a morpheme) surprising, until one considers that transitions between morphemes may not always be clear. Other results become more interesting with some knowledge of Lezgi morphology. The inessive (INESS) and ergative (ERG) case and the oblique stem morpheme (OBL) are identical. The only difference between the first two is the tendency of sentence position, even with Lezgi's free word order. The difference between the latter two is that the ergative morpheme is word final and the the oblique stem is follow by another case morpheme. 
# 
# |__precision__|__recall__|__f1-score__|
# |-----|------|------|
# |0.54|0.49|0.49|
# 
# <center>Table 2: Average score of affix labels only.</center>
# 
# The classifier has most success identifying stem morphemes (STEM) and non-initial letters (I), the majority of which belong to stem morphemes. It has less success with identifying affixes. The classifier is clearly adept at splitting affixes from stems and this is already helpful to human annotators but it would be less helpful splitting strings of affixes and correcly glossing them. Table 2 shows average precision, recall, and f1-score of affix labels is much less accurate than the overall accuracy. This is most likely due in part to homonymic affixes and in part to the fewer instances of affixes compared to stems. As the more texts are correctly annotated with the help of the model, more data can be fed into the training, hopefully increasing the accuracy and incrementally speeding the annotation process.
# 
# The data was also run on a bidirectional sequence-to-sequence deep neural network with attention. The hidden layer size was set at 128, the batch size as 32, the teacher forcing ratio at 0.5. The results in Table 3 indicate that with a small amount of data a supervised classifier can produce equal or better results than a neural network.
# 
# |epochs|accuracy|
# |------|--------|
# |50|0.57|
# |100|0.75|
# |200|0.90|
# |300|0.92|
# |__500__|__0.93__|
# |600|0.89|
# |1000|0.91|
# 
# <center>Table 3: Results of deep neural network</center>

# ## What the Classifier Learned
# 
# By using methods of the crfsuite, we can look insider classifier and see what it learned. From the example printout in Table 3, we can see, for example, that the stem, elative (ELAT), imperfective (IMPF), aorist (AOR), perfective (PF), and plural (PL) morphemes most often consist of more than one letter  but superessive (SUPER), oblique (OBL), and subessive (SUB) morphemes usually consist of just one letter. We can also see that temporal converb (TEMP) morpheme often follows the participle (PTP) morpheme, and another case morpheme tends to follow the oblique, superessive, and subessive case morphemes. These patterns correspond to the facts of Lezgi morphology. On the other hand, both Table 4 and Table 5 indciate that is highly likelythat a  genitive case (GEN) morpheme will be a prefix, which is impossible. This indicates that the affix type (prefix or suffix) might be a useful feature to include.
# 
# |-|-|-|weights|
# |---|---|----|-----|
# |B-SUPER| ->| B-ELAT|  4.820010|
# |B-OBL|  ->| B-SPSS|  3.806645|
# |B-SUB|  ->| B-ELAT|  3.444584|
# |B-OBL|  ->| B-DAT|   2.946830|
# |B-stem| ->| I|       2.258064|
# |B-OBL|  ->| B-GEN|   2.247354|
# |I|      ->| B-OBL|   1.913825|
# |B-stem| ->| B-OBL|   1.862016|
# |B-ELAT| ->| I|       1.711584|
# |B-PTP|  ->| B-TEMP|  1.620690|
# |B-IMPF| ->| I|       1.300227|
# |B-AOR|  ->| I|       1.252594|
# |B-PERF| ->| I|       1.135483|
# |B-PL|   ->| I|       1.043438|
# |B-GEN|  ->| B-stem|  0.956780|
# 
# <center>Table 4: Top most likely transitions</center>
# 
# On the other hand, Table 5, for example, indicates that the negative affix rarely follows a non-initial letter of another morpheme. This is accurate because the negative affix is the only prefix in the language. It is not surprising that the transition still has a greater than zero probability since it is often only one letter long and this letter may be found at the beginning of any word.
# 
# |-|-|-|weights|
# |---|---|---|----|
# |B-ERG|  ->| B-stem|  0.295926|
# |B-TEMP| ->| I|       0.254567|
# |B-SBST| ->| I|       0.249661|
# |I|      ->| B-NEG|   0.221662|
# |B-INF|  ->| B-stem|  0.196340|
# |I|      ->| B-DAT|   0.057729|
# |B-NEG|  ->| B-stem  |0.013683|
# |I|      ->| B-stem|  0.009557|
# |I|      ->| B-ERG|   0.000074|
# |I|      ->| B-SUPER| -0.000692|
# |I|      ->| B-FOC|   -0.003919|
# |I|      ->| B-SBST|  -0.023268|
# |B-OBL|  ->| I|       -0.034257|
# |B-INESS| ->| I|       -0.157967|
# |I|      ->| B-GEN|   -1.180139|
# 
# <center>Table 5: Top most unlikely transitions</center>

# In[174]:

info = tagger.info()

def print_transitions(trans_features):
    '''Print info from the crfsuite.'''
    
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

print("Top likely transitions:")
print_transitions(Counter(info.transitions).most_common(15))

print("\nTop unlikely transitions:")
print_transitions(Counter(info.transitions).most_common()[-15:])


# We can make some observations about the state features. For example, Table 6 indicates that the model rightly recognized that the stem is nearly always at the beginning of the word and there are no consistent feature to identify the non-initial letters of various morphemes. 
# 
# |weight|label|feature|
# |---|---|----|
# |13.385742| B-stem| BOW|
# |6.80475| I|      bias|
# |5.169367| B-PL|   nxt2letters=<ур|
# |5.142534| B-DAT|  letterLowercase=з|
# |4.858094| B-NEG|  letterLowercase=ш|
# |4.568794| B-PTP|  letterLowercase=й|
# |4.513613| B-PST|  letterLowercase=й|
# |4.361416| B-ADSS| letterLowercase=в|
# |4.269127| B-PL|   nxtletter=<р|
# |4.216564| B-FOC|  nxtletter=<и|
# |4.203677| B-GEN|  letterLowercase=н|
# |4.023482| B-INF|  letterLowercase=з|
# |3.977504| B-IMPF| letterLowercase=з|
# |3.868088| B-NEG|  letterLowercase=ч|
# |3.636859| B-FOC|  letterLowercase=н|
# 
# <center>Table 6: Top positive features</center>
# 
# Table 7 indicates that certain letter sequences might be less likely to begin a morpheme. One interesting observation that could be easily confirmed by a corpus study is that the focus particle is least likely to occur on a verb than on any other lexical category. 
# 
# |weight|label|feature|
# |---|----|---|
# |-0.606766| I|      prev2letters=ча>|
# |-0.679616| I|      letterLowercase=ч|
# |-0.704380| I|      prevletter=ш>|
# |-0.741532| I|      prev2letters=ич>|
# |-0.833423| B-FOC|  postag=v|
# |-0.937032| B-FOC|  bias|
# |-1.029693| I|      prev3letters=вал>|
# |-1.071785| I|      nxtletter=<й|
# |-1.073034| I|      prev3letters=гьу>|
# |-1.126576| I|      prev2letters=ди>|
# |-1.150632| B-AOR|  bias|
# |-1.201650| I|      letterLowercase=н|
# |-1.240373| I|      letterLowercase=з|
# |-1.250568| I|      prevletter=р>|
# 
# <center>Table 7: Top negative</center>

# In[175]:

def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-6s %s" % (weight, label, attr))    

print("Top positive:")
print_state_features(Counter(info.state_features).most_common(15))

print("\nTop negative:")
print_state_features(Counter(info.state_features).most_common()[-15:])


# ## Future steps
# 
# The goal of this project was to find a way to speed the work on annotator and improve their accuracy. Since the model reach over the 80% accuracy goal, there seems little reason to try to improve the features, although an examination of the transitions and state features point to a few adjustments that might increase accuracy. The bigggest problem seems to be the almost 50% reduction in predicting the affix glosses. However, the small number of instances found in the test data indicate that this will be improved as the amount of supervised examples increases. The model as it is can speed this increase.
# 
# It should be assumed that few annotators will have programming skills. This is especially true for speakers of minority languages which are often are in areas with limited educational opportunities. The results of this classifier should be checked and corrected by trained annotators. Ideally, this program would be exapnded to write the predicted breaks and glosses to an XML file compatible with FLEx or ELAN or another interface familiar to the annotator or easy to learn. In meantime, the data could be output to an CSV file and presented to the annotator as an spreadsheet.
# 
# Even with carefully annotated training data by a linguist familiar with FLEX and Lezgi morphology, mistakes were made. A few POS tags and affix glosses were missing. This prevents the program from working, but does not tell the user where or what the missing data are. Pre-processing functions should be adjusted so that they present the troublesome morphemes with glosses as a list to the user so that they can be found and corrected using FLEx's concordance feature. 

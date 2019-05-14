# Diego Alberto Barriga Martíenz @umoqnier
# Replica del experimento  'Automatic Prediction of Lezgi Morpheme Breaks' realizado para la lengua Lezgi para generar
# glosa automática a partir de escasos ejemplos

# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
import time
import pycrfsuite
from collections import Counter
from utilities import *
import os

model_filename = 'tsunkua.crfsuite'

# Randomize and split the data

vic_data = get_vic_data()
# data_flex = XMLtoWords("FLExTxtExport2.xml")
aux = WordsToLetter(vic_data)
train_data, test_data = train_test_split(aux, test_size=0.2)

with open("train.txt", "w") as f:
    for t in train_data:
        f.write(str(t) + '\n')

with open("test.txt", "w") as f:
    for t in test_data:
        f.write(str(t) + '\n')



X_train = sent2features(train_data)
Y_train = sent2labels(train_data)

X_test = sent2features(test_data)
Y_test = sent2labels(test_data)

# ### Train the model

trainer = pycrfsuite.Trainer(verbose=False)

for xseq, yseq in zip(X_train, Y_train):    
    trainer.append(xseq, yseq)


# Set training parameters. L-BFGS (what is this) is default. Using Elastic Net (L1 + L2) regularization [ditto?].
trainer.set_params({
        'c1': 1.0,  # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 50  # early stopping
    })


# The program saves the trained model to a file:

if not os.path.isfile(model_filename):
    print("ENTRENANDO...")
    start = time.time()
    trainer.train(model_filename)
    end = time.time()
    print("Fin de entrenamiento. Tiempo de entrenamiento >>", end - start, "[s]", (end - start) / 60, "[m]")
else:
    print("Usando modelo pre-entrenado >>", model_filename)

# ### Make Predictions

tagger = pycrfsuite.Tagger()
print("TAGGER")
tagger.open(model_filename)
print("Fin tagger...")

# First, let's use the trained model to make predications for just one example sentence from the test data.
# The predicted labels are printed out for comparison above the correct labels. Most examples have 100% accuracy.

example_sent = test_data[0]
print('Letters:', '  '.join(extractTokens(example_sent)), end='\n')

print('Predicted:', ' '.join(tagger.tag(extractFeatures(example_sent))))
print('Correct:', ' '.join(extractLabels(example_sent, 1)))


# First, we will predict BIO labels in the test data:

try:
    Y_pred = []
    for i, xseq in enumerate(X_test):
        inter = tagger.tag(xseq)
        Y_pred.append(inter)  # TODO: Resolve critical issue with encoding
except UnicodeDecodeError as e:
    print("UNICODE ERROR AT", i)
    #print("secuencia >> ", X_test[i])
    print("inter >> ", inter)
    print(e.object)
    print(e)



# Get results for labeled position evaluation. This evaluates how well the classifier performed on each morpheme as a
# whole and their tags, rather than evaluating character-level.
# Then, we check the results and print a report of the results. These results are for character level.
eval_labeled_positions(test_decoder(Y_test), Y_pred)


print(bio_classification_report(test_decoder(Y_test), Y_pred))


info = tagger.info()

print("Top likely transitions:")
print_transitions(Counter(info.transitions).most_common(15))

print("\nTop unlikely transitions:")
print_transitions(Counter(info.transitions).most_common()[-15:])


print("Top positive:")
print_state_features(Counter(info.state_features).most_common(15))

print("\nTop negative:")
print_state_features(Counter(info.state_features).most_common()[-15:])
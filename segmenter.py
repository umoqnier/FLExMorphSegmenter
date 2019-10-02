# Diego Alberto Barriga Martíenz @umoqnier
# Replica del experimento  'Automatic Prediction of Lezgi Morpheme Breaks'
# realizado para la lengua Lezgi para generar glosa automática a partir
# de escasos ejemplos

# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
import time
import pycrfsuite
from collections import Counter
from utilities import (get_hard_corpus, get_vic_data, WordsToLetter,
                       sent2features, sent2labels, extractTokens,
                       extractLabels, extractFeatures, eval_labeled_positions,
                       bio_classification_report, labels_decoder,
                       print_state_features, print_transitions, accuracy_score)
import os

corpus_path = 'corpora/'
model_path = 'models/'
model_name = 'hard'
model_filename = f'tsunkua_eval_{model_name}.crfsuite'
debug_mode = False

# Randomize and split the data

vic_data = get_vic_data(corpus_path, 'otomi_original')
train_data, test_data = train_test_split(WordsToLetter(vic_data, False),
                                         test_size=0.2)
hard_corpus = get_hard_corpus(corpus_path, 'hard')
hard_data = WordsToLetter(hard_corpus, True)

if debug_mode:
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

trainer = pycrfsuite.Trainer(verbose=True)

for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)


# Set training parameters. L-BFGS is default. Using Elastic Net (L1 + L2)
# regularization [ditto?].
trainer.set_params({
        'c1': 1.0,  # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 50  # early stopping
    })


# The program saves the trained model to a file:
breakpoint()
if not os.path.isfile(model_path + model_filename):
    print("ENTRENANDO...")
    start = time.time()
    trainer.train(model_path + model_filename)
    end = time.time()
    print("Fin de entrenamiento. Tiempo de entrenamiento >>", end - start,
          "[s]", (end - start) / 60, "[m]")
else:
    print("Usando modelo pre-entrenado >>", model_path + model_filename)

# ### Make Predictions
tagger = pycrfsuite.Tagger()
tagger.open(model_path + model_filename)  # Passing model to tagger

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


# Get results for labeled position evaluation. This evaluates how well
# the classifier performed on each morpheme as a whole and their tags,
# rather than evaluating character-level. Then, we check the results and
# print a report of the results. These results are for character level.
breakpoint()
eval_labeled_positions(y_hard, y_pred_hard)

print(bio_classification_report(y_hard, y_pred_hard))

print("Accuracy Score>>>> ", accuracy_score(y_hard, y_pred_hard))

info = tagger.info()

print("Top likely transitions:")
print_transitions(Counter(info.transitions).most_common(15))

print("\nTop unlikely transitions:")
print_transitions(Counter(info.transitions).most_common()[-15:])


print("Top positive:")
print_state_features(Counter(info.state_features).most_common(15))

print("\nTop negative:")
print_state_features(Counter(info.state_features).most_common()[-15:])


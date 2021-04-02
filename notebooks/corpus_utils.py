import csv
import matplotlib.pyplot as plt
from utils import WordsToLetter, extractFeatures, extractLabels, extractTokens


def obtener_palabras(corpora):
    palabras = []
    for frases in corpora:
        for frase in frases:
            chunks = [palabra[0] for palabra in frase[:-1]]
            palabras.append("".join(chunks))
    return palabras


def obtener_frase(corpora, index):
    palabras = []
    palabra = ''
    frase_etiquetada = corpora[index]
    for etiquetas in frase_etiquetada:
        for texto in etiquetas[:-1]:
            palabra += texto[0]
        palabras.append(palabra)
        palabra = ''
    return " ".join(palabras)


def count_occurrence(bag, _type):
    if _type in bag.keys():
        bag[_type] += 1
    else:
        bag[_type] = 1
    return bag


def dict_sorter(data):
    # Ref https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
    return {k: v for k, v in sorted(data.items(), key=lambda item: item[1], reverse=True)}


def get_tokens(corpus):
    gloss_tags, pos_tags = dict(), dict()
    for frases in corpus:
        for frase in frases:
            for parte in frase[:-1]:
                gloss_tags = count_occurrence(gloss_tags, parte[1])
            pos_tags = count_occurrence(pos_tags, frase[-1])
    return dict_sorter(pos_tags), dict_sorter(gloss_tags)


def get_bio_tokens(corpus):
    bio_tokens = dict()
    frases = WordsToLetter(corpus)
    for frase in frases:
        for letras in frase:
            for letra in letras:
                bio_tokens = count_occurrence(bio_tokens, letra[-1])  # Only Bio Label
    return dict_sorter(bio_tokens)


def graph_maker(data, conf):
    plt.rcParams['figure.figsize'] = [conf['width'], conf['height']]
    # fig, ax = plt.subplot()
    plt.grid()
    plt.xticks(rotation=90)
    # TODO: Cambio en las escalas de los ejes o cambio a barras sin cambiar los ejes
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(conf["xlabel"], fontsize=conf["fontsize"])
    plt.ylabel(conf["ylabel"], fontsize=conf["fontsize"])
    plt.title(conf["title"], fontsize=conf["fontsize"])
    plt.tick_params(axis='both', direction='out', length=5, width=5,
                    labelcolor=conf['labelcolor'], colors=conf['tickscolor'],
                    grid_color=conf['gridcolor'], grid_alpha=0.5)
    if conf["limit"] >= 1:
        plt.plot(list(data.keys())[:conf['limit']],
                 list(data.values())[:conf['limit']],
                 color=conf['color'], linestyle='-.', linewidth=3)
    else:
        plt.plot(list(data.keys()),
                 list(data.values()),
                 color=conf['color'], linestyle='-.', linewidth=3)
    plt.savefig(conf['path'], dpi=300, bbox_inches='tight')
    return plt


def types_to_table(data, columns):
    return "".join([t + " \\\\\n " if i % columns == 0 else t + " & " for i, t in enumerate(data, start=1)])


def tokens_to_table(data, limit=0):
    table = ""
    if limit:
        counts = list(data.items())[:limit]
    else:
        counts = data.items()
    for key, count in counts:
        table += f"{key} & {count} \\\\\n"
    return table


def oto_glosser(word_list, tags, pos_tags=[], corpus_row=0):
    """Funcion que glosa dadas las etiquetas
    
    Esta funcion se encarga de etiquetar una lista de palabras en
    otomi dadas las etiquetas en formato BIO-Label. El etiquetado
    se realiza con la siguiente estructura para cada palabra: 
    
    [[<chunk-word>, <tag>], [<chunk-word>, <tag>], ..., <POS tag>]
    """
    phrase_len, word_len, i = 0, 0, 0
    chunk = ''
    glossed_words, glossed_phrase = [], []
    letters = ''.join(word_list)
    for tag, letter in zip(tags, letters):
        if tag.startswith("B"):
            if chunk:
                try:
                    glossed_words.append([chunk, current_tag[2:]])
                except UnboundLocalError:
                    # TODO: Tratar caso cuando glosa predicha es incorrecta
                    # EJ. Inicia con I-tag y no con B-tag
                    print(f"Wrong BIO-label secuence:")
                    print("BIO-labels:", tags)
                    print("Sentence:", word_list)
                    return [f"ERROR AT {corpus_row}"]
            if len(word_list[i]) == word_len:                
                # Adding POS tag
                #TODO: Adecuarla para que reciba una lista de etiquetas POS
                glossed_words.append(pos_tags[i][-1])
                # Forming phrase structure
                glossed_phrase.append(glossed_words)
                # Next word
                i += 1
                # New word
                word_len = 0
                # New glossed word structure
                glossed_words = []
            chunk = letter
            phrase_len += 1
            word_len += 1
            current_tag = tag
        elif tag.startswith("I"):
            chunk += letter
            phrase_len += 1
            word_len += 1
            if len(letters) == phrase_len:
                glossed_words.append([chunk, current_tag[2:]])
                glossed_words.append(pos_tags[i][-1])
                glossed_phrase.append(glossed_words)
    return glossed_phrase


def words_report(data, letter_corpus, feature_functions_maker, tagger, num_examples=5):
    data = sorted(data, key=lambda t: t[3])
    top = data[:num_examples]
    for i, info in enumerate(top):
        # TODO: Imprimir texto glosado
        print("-"*50)
        index = info[2]
        palabras = info[0]
        example = letter_corpus[index]
        features = feature_functions_maker(example)
        prediction_tags = tagger.tag(features)
        real_tags = extractLabels(example, 1)
        print(f"Ejemplo {i+1} de {num_examples} | Frase: \"{' '.join(palabras)}\" | Accuracy: {info[3]}")
        print("Letra | Predicción | Real | eq?")
        for prediction, real, letter in zip(prediction_tags, real_tags, extractTokens(example)):
            print(f"{letter} | {prediction} | {real} | {True if prediction == real else False}")

            
def gloss_to_csv(phrases, corpus, file_name):
    path = f'ejemplos-csv/{file_name}.csv'
    with open(path, "w") as csvfile:
        header = ["frase", "glosa predicha", "glosa real", "eq?",
                  'accuracy-score']
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for phrase in phrases:
            writer.writerow([" ".join(phrase[0]), str(phrase[1]),
                             corpus[phrase[2]], phrase[1] == corpus[phrase[2]],
                             phrase[3]])
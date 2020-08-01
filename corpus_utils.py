import matplotlib.pyplot as plt
from utils import WordsToLetter, extractFeatures


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

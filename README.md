# CLI

* Se implementa una *Command Line Interface (CLI)* sencilla para mejorar el flujo
de trabajo en los experimentos.

## Instalación
* Haz fork y clona este repositorio
	* `$ git clone https://github.com/<tu-usuario-github>/otomi-morph-segmenter.git`
* Crea un entorno virual de python
	* `$ virtualenv env -p /usr/bin/python3`
	* **NOTA:** Se requiere `python3.6` o mayor
* Activar el entorno
	* `$ source env/bin/activate`
* Instalar las dependencias
	*	`(env)$ pip install -r requeriments.txt`

## Uso

* El CLI se manda llamar de la siguiente manera

```
(env)$ python segmenter.py
```

* Las opciones actuales son las siguientes y se obtienen con el comando
`(env)$ python segmenter --help`

```
Usage: segmenter.py [OPTIONS] HPARAMS

  Command Line Interface para glosador automático del otomí (hñahñu)

Options:
  --corpora-path PATH       Carpeta donde se encuentra el corpus para
                            entrenamiento
  --models-path PATH        Carpeta donde se guardan los modelos entrenados
  -n, --model-name TEXT     Nombre del modelo a entrenar
  -t, --test-size FLOAT     Tamaño del conjunto de pruebas
  -i, --iterations INTEGER  Número máximo de iteraciones de entrenamiento
  -l1 FLOAT                 Parametro de optimización ElasticNet L1
  -l2 FLOAT                 Parametro de optimización ElasticNet L2
  -d, --debug               Habilita el modo depuración
  -v, --verbose             Habilita el modo verboso
  --help                    Show this message and exit.
```

## Cómo correr experimentos

* Los hiperparámetros vienen descritos en el archivo `hyperparams.json`

```json
{
    "name": "base",
    "dataset-train": "corpus_otomi_original",
    "dataset-test": "corpus_hard",
    "max-iter": 50,
    "test-sizes": 0.2,
    "L1": 1.0,
    "L2": 1e-3,
    "description": "Eliminación de la penalización L1. Todo el corpus es de entrenamiento teniendo un corpus solo para evaluación"
}
```
* Las banderas entes mencionadas son opcionales y en caso de existir sobre
escriben los parametros que esten en el archivo `json`
* Si se habilita el modo verboso `-v` se mostrará la información completa de
cada iteración en entrenamiento y reportes del rendimiento del modelo

## Nombre de los modelos

* Con la intención de tener documentada la evolución de los experimentos el
nombre de los modelos sigue la siguiente estructura
* `tsu_<nombre>_<max-iteraciones>_<L1>_<L2>.crfsuite`
* **Estos y otros parametros son configurables via `hyperparams.json`**

# Plan de trabajo

* [x] Revisar los reportes que imprime el programa y como esta evaluando
* [ ] Documentación de funciones
* [x] Anotar para cada experimento
    * Nombre del modelo
    * Máximo de Iteraciones
    * Test size
    * Data sets
    * Elastic Net l1, l2
    * Accuracy
    * Descripción
* [x] Agregar una seccion de parametros para no hardcodear
* [ ] Correr experimentos con diferentes parámetros

## Modificación de parametros

1. Quitar todas las etiquetas POS y entrenar. Verificar
2. Quitar la ventana de contexto y limitarla a 3
3. Que L1 sea 0 y que L2 sea 0 y ambos para ver que sucede
4. Modificar el split ratio

# Escritura

## Metodología - Revision de la literatura

1. Dominar CRF
    * Saber que son y como se definen
        * feature functions
        * como se defina la funcion de probabilidad
    * Que parametros de optimizacion aceptan
        * Metodo de optimizacion
        * Regularizacion L1 y L2
        * Buscar más si tiene
2. Hablar de otomi
    * Tipo de lengua
    * Donde se habla
    * Importancia de tratarlo

## Estado del arte

1. Articulos que usan CRF para lenguas de bajos recursos
    * Griko
    * Lezgi

## Descripcion de los experimentos

* Interpretar las medidas de evaluacion
* Explicar por que con la modificacion de ciertos parametros aumenta o
disminuye

## Depuración del corpus
* Para eliminar líneas repetidas en el corpus `$ sort -u corpus > corpus_uniq`
* Mostrar las líneas duplicadas `$ diff --color corpus_sort corpus_uniq`

# Convenciones

## Sustitución de letras
Para solucionar problemas de encoding/decoding se sustituyen las siguientes
letras en otomí:

* u̱ -> μ
* a̱̱ -> α
* e̱ -> ε
* i̱ -> ι


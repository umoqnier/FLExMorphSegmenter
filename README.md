# CLI

* Se implementa una *Command Line Interface (CLI)* sencilla para mejorar el
flujo de trabajo en los experimentos.

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
    "k": 0,
    "description": "Eliminación de la penalización L1. Todo el corpus es de entrenamiento teniendo un corpus solo para evaluación"
}
```
* Las banderas entes mencionadas son opcionales y en caso de existir sobre
escriben los parametros que esten en el archivo `.json`
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
    * K para K folds
    * Descripción
* [x] Agregar una seccion de parametros para no hardcodear
* [x] Correr experimentos con diferentes parámetros
* [x] Correr adaptación de lezgi con el CLI
* [x] Correr K folds con K = 10
* [x] Correr con parametros de penalización = 0
* [ ] Obtener el vocabulario de etiquétas

## Modificación de *feature functions*

1. [x] Quitar todas las etiquetas POS
2. [ ] Quitar la ventana de contexto y limitarla a 3
3. [x] Quitar la ventana de contexto y limitarla a 1 simulando un HMM

## *Baseline*

* Feature functions solo con la letra anterior
  * Sin regularización L1 y L2
  * Simulando HMM
  * Nombre del modelo: *`tsu_baseline_zero_l1_l2_test_30%_50_0_0.crfsuite`*

### Esperimentos finales

* Trabajar con todas las feature functions
  * K fold = 10
  * [x] L1 = 0.1 y L2 = 0.001 *`tsu_base_k_fold_50_0.1_0.001_k_10.crfsuite`*
  * [x] L1 = 0 *``*
  * [x] L2 = 0 *``*
  * [x] L1 = 0 y L2 = 0 *`tsu_l1_l2_zero_k_fold_10_50_0_0_k_10.crfsuite`*
* Trabajar con todas las las feature functions excepto POS
  * K fold = 10
  * [x] L1 = 0.1 y L2 = 0.001
  * [x] L1 = 0
  * [x] L2 = 0
  * [x] L1 = 0 y L2 = 0

## Graficas

* Graficar la funcion de perdida
  * Guardar la información y recuperar la de los modelos previos

## Manejo de Datos para evaluación

1. Juntar todo: Un solo dataset y hacer validación cruzada (k-fold cross validation)
	* Dataset (original) + instancias nuevas ("hard" etiquetadas por Vic)
	* [x] K=5
	* [x] K=10
	* El accuracy es el promedio de todas las particiones de como lo está haciendo
2. Hold out evaluation (Actual): Se aparta una partición de datos para test
	* Data set original
	* Reducir la partición de entrenamiento
		* [x] 80%(train) - 20%(test) + hard
		* [x] 70%(train) - 30%(test) + hard
		* [x] 75%(train) - 25%(test) + hard

## Decisiones experimentales restantes

* Análisis de features
* Baseline (posibles):
	* Parametros que funcionaron el Lezgi probados en el otomí
	* Otros métodos
		* HMM
		* SVM
		* Red Neuronal seq2seq

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

# Trabajando con el base line

## Baseline
Sin regularización L1 y L2 y feature functions solo con la letra anterior (simulando HMM).
Nombre del modelo: *`tsu_baseline_zero_l1_l2_test_30%_50_0_0.crfsuite`*

* Trabajar con todas las feature functions
  * K fold = 10
  * L1 = 0
  * L2 = 0
  * L1 = 0 y L2 = 0
* Trabajar con todas las las feature functions excepto POS
  * K fold = 10
  * L1 = 0
  * L2 = 0
  * L1 = 0 y L2 = 0

## Agregar más feature functions??

## Graficas

* Graficar la funcion de perdida

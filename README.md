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
* Las banderas antes mencionadas son opcionales y en caso de existir sobre
escriben los parametros que esten en el archivo `.json`
* Si se habilita el modo verboso `-v` se mostrará la información completa de
cada iteración en entrenamiento y reportes del rendimiento del modelo

## Nombre de los modelos

* Con la intención de tener documentada la evolución de los experimentos el
nombre de los modelos sigue la siguiente estructura
* `tsu_<nombre>_<max-iteraciones>_<L1>_<L2>.crfsuite`
* **Estos y otros parametros son configurables via `hyperparams.json`**


## Modificación de *feature functions*

1. [x] Quitar todas las etiquetas POS
2. [ ] Quitar la ventana de contexto y limitarla a 3
3. [x] Quitar la ventana de contexto y limitarla a 1 simulando un HMM

## *Baseline*

* Feature functions solo con la letra anterior
  * Sin regularización L1 y L2
	* Máximo de iteraciones 50
  * Simulando HMM
  * Nombre del modelo: *`tsu_baseline_HMMLike_zero_50_0_0_[1-10].crfsuite`*


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

## *Pipeline*

1. Obtención del copus glosado
2. Preprocesamiento del texto
3. Creación de las *feature functions*
4. Separación en conjunto de entrenamiento y pruebas
5. Fase de entrenamiento y construcción del modelo
6. Generación de etiquetas y pruebas de rendimiento

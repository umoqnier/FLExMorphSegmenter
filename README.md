# Plan de trabajo

* [x] Revisar los reportes que imprime el programa y como esta evaluando
* [ ] Documentación de funciones
* [x] Anotar para cada experimento
    * Nombre del modelo
    * Máximo de Iteraciones
    * Split ratio
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

# Cómo correr experimentos

* TODO

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

## Nombre de los modelos
Con la intención de tener documentada la evolución de los experimentos el
nombre de los modelos sigue la siguiente estructura
* `tsu_<nombre>_<max-iteraciones>_<L1>_<L2>.crfsuite`
	* *Estos y otros parametros son configurables via `hyperparams.json`* 

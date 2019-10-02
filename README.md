# Instrucciones para experimentos

* Para correr el programa se requiere `python3.x` y el manejador de dependencias `pip`
* Instalar las dependencias con `pip install -r requeriments.txt`
* Correr el programa con `python segmenter.py`
    * Al correr el programa se genera un modelo de aprendizaje. El nombre de dicho modelo se modifica en la variable`model_filename` 

 **NOTA**: El corpus utilizada se encuentra en el archivo `corpusOtomi.txt`

# Plan de trabajo

1. Revisar los reportes y como esta evaluando
2. Anotar para cada experimento
    * Nombre del modelo
    * Elastic Net l1, l2
    * Feature function
    * Iteraciones
    * Accuracy
3. Agregar una seccion de parametros para no hardcodear

## Modificaciones al experimento actual

1. Quitar todas las etiquetas POS y entrenar. Verificar
2. Quitar la ventana de contexto y limitarla a 3
3. Que L1 sea 0 y que L2 sea 0 y ambos para ver que sucede

## Escritura

### Metodología - Revision de la literatura

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

### Estado del arte

1. Articulos que usan CRF para lenguas de bajos recursos
    * Griko
    * Lezgi

### Descripcion de los experimentos

* Interpretar las medidas de evaluacion
* Explicar por que con la modificacion de ciertos parametros aumenta o disminuye 

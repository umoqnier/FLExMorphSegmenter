# Tareas por hacer (y hechas)

## Texto

* [ ] Introducción
* [ ] Marco teórico
* [ ] Metodología
* [ ] Resultados
	* [ ] Análisis
		* [ ] Mencionar la eficiencia por etiqueta de la tabla 4.4
		* [ ] Ejemplos de etiquetados de los modelos de la tabla 4.4
	* [ ] Gráficas de la función de perdida de los modelos de la tabla 4.4 de k=1 y k=10
		* Comentar la gráfica
* [ ] Conclusiones

## Notebooks

* [ ] Creación de notebooks de jupyter para extracción de información del corpus
	* [x] Ordenar los notebooks
		* Adecuar funcionamiento
	* Tipos y tokens
	* Distribución de etiquetas
	* Conteo de palabras
	* Tamaño del corpus
	* Longitud de palabras promedio
	* Distribución de frecuencias basadas en la longitud de las palabras
* [ ] Ejemplos de etiquetado
* [x] Función de perdida

## Graficas

* [x] Guardar la información y recuperar la de los modelos previos
* [ ] Extractor automático de los datos
* [ ] Gráfica de la función de perdida

## Plan de trabajo

* [ ] Documentación de funciones
* [x] Revisar los reportes que imprime el programa y como esta evaluando
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
* [X] Obtener el vocabulario de etiquétas

## Decisiones experimentales restantes

* [ ] Análisis de features
* [x] Baseline (posibles):
	* [x] Parametros que funcionaron el Lezgi probados en el otomí
	* [-] Otros métodos
		* [x] HMM (Simulado con CRFs)
		* [ ] SVM
		* [ ] Red Neuronal seq2seq

### Esperimentos finales

* [x] Trabajar con todas las feature functions
  * [x] K fold = 10
  * [x] L1 = 0.1 y L2 = 0.001 *`tsu_base_k_fold_50_0.1_0.001_k_10.crfsuite`*
  * [x] L1 = 0 
  * [x] L2 = 0
  * [x] L1 = 0 y L2 = 0 *`tsu_l1_l2_zero_k_fold_10_50_0_0_k_10.crfsuite`*
* [x] Trabajar con todas las las feature functions excepto POS
  * [x] K fold = 10
  * [x] L1 = 0.1 y L2 = 0.001
  * [x] L1 = 0
  * [x] L2 = 0
  * [x] L1 = 0 y L2 = 0


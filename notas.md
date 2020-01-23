# Notas para hacer una tesis genial

## Metodología

### CRF

* [Modelo gráfico a utilizar](http://www.chokkan.org/software/crfsuite/manual.html#idp8849121424): 
TODO: Investigar esto
The 1st-order Markov CRF with state and transition features (dyad features).
State features are conditioned on combinations of attributes and labels, and
transition features are conditioned on label bigrams.

* [Algorítmo de aprendizaje](http://www.chokkan.org/software/crfsuite/manual.html#idp8849121424): 
lbfgs: Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) method.

Maximizar el logaritmo de la verosimilitud de los datos de entrenamiento con
los términos de regulación L1 y/o L2 usando el método L-BFGS. Cuando un
coeficiente L1 es especificado en cero, el algoritmo cambia al método
Orthant-Wise Limited-memory Quasi-Newton (OWL-QN). Prácticamente, este algoritmo
mejora los pesos de características muy lento al inicio del proceso de 
entrenamiento, pero converge a los pesos de características optimos rápidamente
al final

  * Hiperparámetros: 
    * c1=VALUE The coefficient for L1 regularization. If a non-zero value is
      specified, CRFsuite switches to the Orthant-Wise Limited-memory
      Quasi-Newton (OWL-QN) method. The default value is zero (no L1
      regularization).
    * c2=VALUE The coefficient for L2 regularization. The default value is 1.
    * max_iterations=NUM The maximum number of iterations for L-BFGS
      optimization. The L-BFGS routine terminates if the iteration count
      exceeds this value. The default value is set to the maximum value of
      integer on the machine (INT_MAX).
    * num_memories=NUM The number of limited memories that L-BFGS uses for
      approximating the inverse hessian matrix. The default value is 6.
    * epsilon=VALUE The epsilon parameter that determines the condition of
      convergence. The default value is 1e-5.
    * stop=NUM The duration of iterations to test the stopping criterion. The
      default value is 10.
    * delta=VALUE The threshold for the stopping criterion; an L-BFGS iteration
      stops when the improvement of the log likelihood over the last \${stop}
      iterations is no greater than this threshold. The default value is 1e-5.
    * linesearch=STRING The line search method used in the L-BFGS algorithm.
      Available methods are: "MoreThuente" (MoreThuente method proposd by More
      and Thuente), "Backtracking" (Backtracking method with regular Wolfe
      condition), and "StrongBacktracking" (Backtracking method with strong
      Wolfe condition). The default method is "MoreThuente".
    * max_linesearch=NUM The maximum number of trials for the line search
      algorithm. The default value is 20.

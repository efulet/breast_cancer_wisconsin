#Estudio comparativo de las tecnicas de clasificacion sobre el cancer de mama

##Problema

Se desea construir un sistema que sea capaz de diagnosticar la presencia de 
cancer mamario. Se dispone de un conjunto de datos denominado Breast Cancer 
Wisconsin (Diagnostic) Data Set. El conjunto de datos, junto a su descripción, 
se puede descargar desde http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

##Desarrollo

El trabajo consiste en construir un modelo de regresion lineal y un modelo 
rectangular para predecir si un paciente tiene cancer mamario o no. El modelo 
de regresion lineal ya se encuentra implementado en python (sklearn). El modelo 
rectangular ha sido implementado en este paquete. Se utilizara un conjunto de 
datos para la construccion y validacion de los modelos. Los pasos para el 
desarrollo del experimento son los siguientes:

  1. Carga de los datos
  2. Seleccion de datos para el entrenamiento del modelo y pruebas
  3. Construccion del modelo de regresion lineal
  4. Reportar de los resultados encontrados
  5. Construir un clasificador utilizando el modelo rectangular propuesto en clases
  6. Reportar de los resultados encontrados

##Como ejecutar el programa

NOTA: Se requiere python version 2.7

Este programa puede ejecutarse de la siguiente manera:

```bash
  $> python wdbc/main.py
```

##Salida

```bash
********************************************************************************
Data set dimension: 30
N instances data set: 569
N instances training set: 398
N instances testing set: 171
********************************************************************************
********************************************************************************
Modelo Lineal
********************************************************************************
Linear regression fit time: 0:00:00.010694
Linear regression predict time: 0:00:00.000068
Apparent error rate: 0.042714
Mean absolute error: 0.0585
Confusion matrix:
[[108   0]
 [ 10  53]]
Mean square error: 0.0585
Accuracy score: 0.9415
The coefficient of determination: 0.7487
Recall(macro): 0.8413
Sensitivity:
[ 0.84126984  1.        ]
Specificity:
[ 0.  1.]
[class=0] F_{beta}(1): 0.9558
[class=0] Support: 108.0000
[class=1] F_{beta}(1): 0.9138
[class=1] Support: 63.0000
********************************************************************************
Modelo Rectangular
********************************************************************************
Rectangular classifier fit time: 0:00:00.050399
Rectangular classifier predict time: 0:00:00.072698
Apparent error rate: 0.153266
Mean absolute error: 0.2047
Confusion matrix:
[[84 24]
 [11 52]]
Mean square error: 0.2047
Accuracy score: 0.7953
The coefficient of determination: 0.1204
Recall(macro): 0.8254
Sensitivity:
[ 0.          0.82539683  1.        ]
Specificity:
[ 0.          0.22222222  1.        ]
[class=0] F_{beta}(1): 0.8276
[class=0] Support: 108.0000
[class=1] F_{beta}(1): 0.7482
[class=1] Support: 63.0000
```

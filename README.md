# Fran_Arenas_XAI_credit_card

En este repositorio se busca aplicar herramientas de XAI (Explainable AI) sobre un modelo de predicción de crédito para probar la validez de dichas técnicas.


Para ello se utilizará el dataset encontrado en: https://www.kaggle.com/rikdifos/credit-card-approval-prediction?select=application_record.csv


La principal herramienta para aplicar la explicabilidad de nuestro modelo se trata de la librería SHAP basada principalmente en el Shapley value utilizado en teoría de juegos: https://shap.readthedocs.io/en/latest/index.html

## Dataset

Dispones de dos datasets. En el primero se registran los datos del cliente para una solicitud y en el siguiente se registra el siguimiento de la deuda del cliente.


Para predecir los clientes potencialmente peligrosos se han fusionado los dos dataset para añadir a los datos del cliente la variable de cliente fraudulento que será utilizada como variable objetivo.


Algunos de los datos del ciente son : género, vehículo propio, número de hijos, educación, estatus familiar, tipo de domicilio, tipo de trabajo, etc.

![](images/clientspng.png)

![](images/jobs.png)

![](images/Male_Female.png)

![](images/education.png)




## Modelo


Se ha decidido utilizar el modelo XGBoost. Debido al desbalanceamiento del dataset (corregido parcialmente con la ténica SMOTE) y a la falta de poder computacional para realizar "hyperparameter tunning" no se han obtenido unos resultados demasiado prometedores, pero al no ser la precisión un requisito para la prueba de las técnicas de explicabilidad utilizadas serà una información suficiente.


## Explicabilidad del modelo


Lorem Ipsum Dolor Sit Amet

![](images/Importance.png)


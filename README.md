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

En este notebook utilizo principalmente la librería shap

### Explicabilidad en un individuo
 Para ello se puede ver el impacto de cada variable (según los Shapley Values) en la toma de decisión de nuestro modelo. En este caso observamos una muestra de un usuario que ha sido clasificado como cliente de alto riesgo. Se pueden graficar tantas variables como las disponibles del cliente, pero para una visualización más cómoda de los datos solo se han graficado las más relevantes.
 
![](images/Importance.png)

### Explicabilidad para múltiples individuos de manera simultánea
SHAP ofrece la posibilidad de graficar de manera vertical el gráfico anterior creando así una visualización simultánea en la que se pueden detectar visualmente el impacto de multiples variables en nuestro modelo. Esta herramienta es interactiva, por lo que en apartado de recursos se encuentra un archivo html con el cual se puede visualizar información diversa. 


El principal problema de esta herramienta es que tiene un coste computacional elevado, por lo que representar un número significativo de muestras será necesario disponer de un hardware adecuado. En este caso se han utilizado 2000 muestras (de un total de 25134).


Es útil principalmente para variables con valores continuos, pero debido a los datos de los cuales disponemos la mayoría de las capturas interesantes son de valores discretos.


Algunas capturas de pantalla interesantes sobre la gráfica:


CAPTURA 1


Aquí podemos ver como la variable hijos tiene un impacto superior en mujeres que en hombres (esto se debe probablemente a que las mujeres con 0 hijos tengan un impacto fuerte sobre la decisión, por lo que la relación entre ambas variables se incrementa)


CAPTURA 2

En el gráfico anterior se puede observar como a la edad de 54 años se observa un pico en el impacto con la variable años trabajados. Al ser tan acentuado en los 54 años en comparación con otros años cercanos nos indica que probablemente sea un sesgo del modelo.


### Variables con más impacto en el modelo

impact.png

En el gráfico anterior se puede observar rápidamente las variables con un mayor impacto en el modelo, aunque se puede hacer un estudio más exaustivo de los valores de Shapley expuesto en la siguiente sección.

### Valores de Shapley y detección de sesgos
Mediante una función propia del notebook asociado a este repositorio se ha mostrado el impacto de cada valor posible de cada "feature" de nuestro dataset.

Feature: CODE_GENDER	 value: 1	 Impact: -0.14596430979669409
Feature: CODE_GENDER	 value: 0	 Impact: -0.23019137926356806
---

Feature: FLAG_OWN_CAR	 value: 1	 Impact: -0.1749546868327881
Feature: FLAG_OWN_CAR	 value: 0	 Impact: -0.06634596839147225


Feature: FLAG_OWN_REALTY	 value: 1	 Impact: -0.29088786100174524
Feature: FLAG_OWN_REALTY	 value: 0	 Impact: -0.17027427767716796


Feature: CNT_CHILDREN	 value: 0	 Impact: -0.025556012098349824
Feature: CNT_CHILDREN	 value: 3	 Impact: -0.024545736931777824
Feature: CNT_CHILDREN	 value: 1	 Impact: -0.13874566251813417
Feature: CNT_CHILDREN	 value: 2	 Impact: -0.1214861305387653
Feature: CNT_CHILDREN	 value: 4	 Impact: -0.10611623264195626
Feature: CNT_CHILDREN	 value: 14	 Impact: -0.1476748287677765
Feature: CNT_CHILDREN	 value: 5	 Impact: -0.22494369993607202
Feature: CNT_CHILDREN	 value: 19	 Impact: 0.13113021850585938
Feature: CNT_CHILDREN	 value: 7	 Impact: 0.30773746967315674


Feature: NAME_EDUCATION_TYPE	 value: 0	 Impact: -0.17723326800877512
Feature: NAME_EDUCATION_TYPE	 value: 1	 Impact: -0.15397021899685898
Feature: NAME_EDUCATION_TYPE	 value: 2	 Impact: -0.1648118817354135
Feature: NAME_EDUCATION_TYPE	 value: 3	 Impact: -0.19202636333834877
Feature: NAME_EDUCATION_TYPE	 value: 4	 Impact: 0.31512815185955595


Feature: NAME_FAMILY_STATUS	 value: 0	 Impact: -0.15527148092126972
Feature: NAME_FAMILY_STATUS	 value: 1	 Impact: -0.15773843879276192
Feature: NAME_FAMILY_STATUS	 value: 2	 Impact: -0.15378473917613103
Feature: NAME_FAMILY_STATUS	 value: 3	 Impact: -0.15636435477991525
Feature: NAME_FAMILY_STATUS	 value: 4	 Impact: -0.16758495444665986


Feature: NAME_HOUSING_TYPE	 value: 0	 Impact: -0.11141496778693058
Feature: NAME_HOUSING_TYPE	 value: 1	 Impact: -0.07254216297060795
Feature: NAME_HOUSING_TYPE	 value: 2	 Impact: -0.09393835136516679
Feature: NAME_HOUSING_TYPE	 value: 3	 Impact: -0.11447659513458831
Feature: NAME_HOUSING_TYPE	 value: 4	 Impact: -0.19556258770784266
Feature: NAME_HOUSING_TYPE	 value: 5	 Impact: -0.11665457103345263


Feature: OCCUPATION_TYPE	 value: 0	 Impact: -0.03349234885133007
Feature: OCCUPATION_TYPE	 value: 1	 Impact: -0.05179758163093052
Feature: OCCUPATION_TYPE	 value: 2	 Impact: -0.06476086363801245
Feature: OCCUPATION_TYPE	 value: 3	 Impact: -0.04191095964220474
Feature: OCCUPATION_TYPE	 value: 4	 Impact: -0.04662614272729236
Feature: OCCUPATION_TYPE	 value: 5	 Impact: -0.0423098126836876
Feature: OCCUPATION_TYPE	 value: 6	 Impact: -0.04515066756433435
Feature: OCCUPATION_TYPE	 value: 7	 Impact: -0.04451314151542637
Feature: OCCUPATION_TYPE	 value: 8	 Impact: -0.05025064239313086
Feature: OCCUPATION_TYPE	 value: 9	 Impact: -0.08514854659767669
Feature: OCCUPATION_TYPE	 value: 10	 Impact: -0.05862337914337706
Feature: OCCUPATION_TYPE	 value: 11	 Impact: -0.006659073430512632
Feature: OCCUPATION_TYPE	 value: 12	 Impact: -0.0564259551007976
Feature: OCCUPATION_TYPE	 value: 13	 Impact: -0.0586501101204595
Feature: OCCUPATION_TYPE	 value: 14	 Impact: -0.015111457010538414
Feature: OCCUPATION_TYPE	 value: 15	 Impact: -0.0713506354840801
Feature: OCCUPATION_TYPE	 value: 16	 Impact: -0.07102132768995023
Feature: OCCUPATION_TYPE	 value: 17	 Impact: -0.082166072834904


Feature: YEARS_BIRTH	 value: 58	 Impact: -0.015574779790402308
Feature: YEARS_BIRTH	 value: 52	 Impact: -0.02285164684348761
Feature: YEARS_BIRTH	 value: 46	 Impact: -0.012326259708608085
Feature: YEARS_BIRTH	 value: 48	 Impact: -0.02894945775971066
Feature: YEARS_BIRTH	 value: 29	 Impact: -0.028600397832459133
Feature: YEARS_BIRTH	 value: 27	 Impact: -0.03445303100870883
Feature: YEARS_BIRTH	 value: 34	 Impact: -0.030068792898197298
Feature: YEARS_BIRTH	 value: 32	 Impact: -0.024724324926545638
Feature: YEARS_BIRTH	 value: 56	 Impact: -0.018883141433304575
Feature: YEARS_BIRTH	 value: 43	 Impact: -0.01911626078261734
Feature: YEARS_BIRTH	 value: 44	 Impact: -0.028282802675886037
Feature: YEARS_BIRTH	 value: 45	 Impact: -0.02629918771572411
Feature: YEARS_BIRTH	 value: 42	 Impact: 0.001169054145057624
Feature: YEARS_BIRTH	 value: 37	 Impact: -0.016611816522215023
Feature: YEARS_BIRTH	 value: 57	 Impact: -0.022651242536346917
Feature: YEARS_BIRTH	 value: 51	 Impact: -0.019604974981709
Feature: YEARS_BIRTH	 value: 54	 Impact: -0.019881296712975786
Feature: YEARS_BIRTH	 value: 39	 Impact: -0.016374802775986512
Feature: YEARS_BIRTH	 value: 28	 Impact: -0.023660210526957326
Feature: YEARS_BIRTH	 value: 30	 Impact: -0.031567678681821545
Feature: YEARS_BIRTH	 value: 24	 Impact: -0.000263988900976845
Feature: YEARS_BIRTH	 value: 20	 Impact: -0.028132319450378418
Feature: YEARS_BIRTH	 value: 38	 Impact: -0.02475334024507742
Feature: YEARS_BIRTH	 value: 40	 Impact: -0.02297415497161972
Feature: YEARS_BIRTH	 value: 33	 Impact: -0.02865118888956135
Feature: YEARS_BIRTH	 value: 36	 Impact: -0.016914591162586784
Feature: YEARS_BIRTH	 value: 35	 Impact: -0.02491063135533245
Feature: YEARS_BIRTH	 value: 41	 Impact: -0.0243860163756378
Feature: YEARS_BIRTH	 value: 26	 Impact: -0.028617619238036865
Feature: YEARS_BIRTH	 value: 53	 Impact: -0.02186263954476201
Feature: YEARS_BIRTH	 value: 50	 Impact: -0.0194116448249963
Feature: YEARS_BIRTH	 value: 25	 Impact: -0.03073999951356378
Feature: YEARS_BIRTH	 value: 22	 Impact: 0.03690473295497794
Feature: YEARS_BIRTH	 value: 23	 Impact: -0.014780475570739287
Feature: YEARS_BIRTH	 value: 47	 Impact: -0.019356126880877515
Feature: YEARS_BIRTH	 value: 31	 Impact: -0.03577966180421738
Feature: YEARS_BIRTH	 value: 49	 Impact: -0.023178530240991285
Feature: YEARS_BIRTH	 value: 59	 Impact: -0.019644704287260076
Feature: YEARS_BIRTH	 value: 60	 Impact: -0.024906580221935516
Feature: YEARS_BIRTH	 value: 55	 Impact: -0.011394077953365113
Feature: YEARS_BIRTH	 value: 62	 Impact: -0.01771018660568458
Feature: YEARS_BIRTH	 value: 61	 Impact: -0.007766950050490382
Feature: YEARS_BIRTH	 value: 64	 Impact: -0.013110409705148695
Feature: YEARS_BIRTH	 value: 21	 Impact: 0.004582584649324417
Feature: YEARS_BIRTH	 value: 63	 Impact: -0.007372539079583743
Feature: YEARS_BIRTH	 value: 65	 Impact: -0.00960556510835886
Feature: YEARS_BIRTH	 value: 66	 Impact: -0.010364395307583941
Feature: YEARS_BIRTH	 value: 67	 Impact: -0.028812579810619354


Feature: YEARS_EMPLOYED	 value: 3	 Impact: -0.08185836481326786
Feature: YEARS_EMPLOYED	 value: 8	 Impact: -0.08631414426219644
Feature: YEARS_EMPLOYED	 value: 2	 Impact: -0.08838177915041297
Feature: YEARS_EMPLOYED	 value: 4	 Impact: -0.08162173346461822
Feature: YEARS_EMPLOYED	 value: 5	 Impact: -0.09856720255930641
Feature: YEARS_EMPLOYED	 value: 12	 Impact: -0.1010800766065402
Feature: YEARS_EMPLOYED	 value: 19	 Impact: -0.0654479770118831
Feature: YEARS_EMPLOYED	 value: 14	 Impact: -0.06692343382608323
Feature: YEARS_EMPLOYED	 value: 13	 Impact: -0.06627055069948178
Feature: YEARS_EMPLOYED	 value: 6	 Impact: -0.0903734313743567
Feature: YEARS_EMPLOYED	 value: 17	 Impact: -0.10845855865168424
Feature: YEARS_EMPLOYED	 value: 29	 Impact: -0.07050702581182122
Feature: YEARS_EMPLOYED	 value: 7	 Impact: -0.08717696989789236
Feature: YEARS_EMPLOYED	 value: 1	 Impact: -0.08810607130348348
Feature: YEARS_EMPLOYED	 value: 15	 Impact: -0.08465731380835333
Feature: YEARS_EMPLOYED	 value: 11	 Impact: -0.09649126927638095
Feature: YEARS_EMPLOYED	 value: 0	 Impact: -0.08738830309108192
Feature: YEARS_EMPLOYED	 value: 10	 Impact: -0.077264581531109
Feature: YEARS_EMPLOYED	 value: 23	 Impact: -0.08503892017609399
Feature: YEARS_EMPLOYED	 value: 24	 Impact: -0.09238022353653748
Feature: YEARS_EMPLOYED	 value: 20	 Impact: -0.06598485276079834
Feature: YEARS_EMPLOYED	 value: 9	 Impact: -0.07621552821004775
Feature: YEARS_EMPLOYED	 value: 26	 Impact: -0.08209180018554131
Feature: YEARS_EMPLOYED	 value: 18	 Impact: -0.08134186200925476
Feature: YEARS_EMPLOYED	 value: 21	 Impact: -0.06538435519110117
Feature: YEARS_EMPLOYED	 value: 22	 Impact: -0.07378811930069086
Feature: YEARS_EMPLOYED	 value: 16	 Impact: -0.09518834722170419
Feature: YEARS_EMPLOYED	 value: 28	 Impact: -0.08201357819139958
Feature: YEARS_EMPLOYED	 value: 27	 Impact: -0.06899432390135654
Feature: YEARS_EMPLOYED	 value: 38	 Impact: -0.024758146454890568
Feature: YEARS_EMPLOYED	 value: 32	 Impact: -0.11922794930567895
Feature: YEARS_EMPLOYED	 value: 31	 Impact: -0.1421568856375026
Feature: YEARS_EMPLOYED	 value: 36	 Impact: -0.1902722823433578
Feature: YEARS_EMPLOYED	 value: 37	 Impact: -0.03051195596344769
Feature: YEARS_EMPLOYED	 value: 30	 Impact: -0.04681832396558353
Feature: YEARS_EMPLOYED	 value: 39	 Impact: -0.039605066857554695
Feature: YEARS_EMPLOYED	 value: 34	 Impact: -0.05805595007471063
Feature: YEARS_EMPLOYED	 value: 25	 Impact: -0.06566558142243199
Feature: YEARS_EMPLOYED	 value: 33	 Impact: -0.07431342948058789
Feature: YEARS_EMPLOYED	 value: 40	 Impact: -0.02477440855000168
Feature: YEARS_EMPLOYED	 value: 35	 Impact: -0.08101052670709548
Feature: YEARS_EMPLOYED	 value: 43	 Impact: -0.005770289804786444
Feature: YEARS_EMPLOYED	 value: 42	 Impact: -0.09047297388315201



### Gráficas de interacción de impacto entre variables 

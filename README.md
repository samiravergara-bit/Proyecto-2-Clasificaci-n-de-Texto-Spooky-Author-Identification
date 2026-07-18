# Proyecto 2 Clasificación de texto Spooky Author Identification

**Autora:** Samira Vergara

## Descripción del problema
El objetivo de este proyecto es resolver un problema de clasificación de texto, la idea es identificar a cuál de tres autores de literatura de terror pertenece un fragmento de texto u oración en especifica. Los autores a clasificar son:

* EAP: Edgar Allan Poe
* HPL: H.P. Lovecraft
* MWS: Mary Wollstonecraft Shelley

## Dataset
El dataset fue extraído de la competencia de Kaggle *"Spooky Author Identification"* (https://www.kaggle.com/competitions/spooky-author-identification/data). Consiste en miles de frases extraídas de las obras de los tres autores. El archivo principal ("train.csv") incluye un identificador único, el texto original (frase en inglés) y la etiqueta categórica del autor(las que enlisté en un inicio). El dataset original se encuentra balanceado (a diferencia del práctico de spam donde dominaba una clase), sobresaliendo levemente EAP. Por esto mismo utilicé acurracy como métrica principal complementada con F1-macro.
Usé un 80% de entrenamiento y 20% de test. De ese 80% saqué de manera adicional un 10% como conjunto de validación (usado para early stopping de BERT), así quedaron ambos modelos (Naive Bayes y BERT) entrenados sobre exactamente el mismo 90% de los datos para que la comparación entre ellos sea justa.

## Justificación de los modelos seleccionados

Decidí comprar 2 enfoques de naturaleza distinta para contrastar sus ventajas, limitaciones y pertinencia frente al dataset porque la pauta me pedía una discusión critica de los resultados entonces necesitaba algo con que comparar para saber si iba bien o mal.

1. Naive Bayes: Lo elegí como mi modelo de control. Como vimos en clases, es un modelo de clasificacion diseñado especificamente para conteos de palabras (bag of words), que es justo lo que hago en este problema. Es rápido de entrenar y evaluar, con muy pocos parámetros en relación a la cantidad de datos disponibles lo que lo hace poco propenso a memorizar (overfitting). La limitación es que no captura relaciones de estructura y ritmo, así que las coincidencias que encuentra es más limitada.

2. BERT (bert-base-uncased): Para el enfoque de Deep Learning, elegí este modelo pre-entrenado sobre texto en inglés. A diferencia de Naive Bayes, BERT sí lee la oración completa y entiende el contexto. Algunas ventajas son que al usar Transfer Learning (congelando su conocimiento previo) reduje el costo computacional y así aproveché su conocimiento previo, también reduje el riesgo de overfitting. La limitación es que al estar congelado, Bert no se ajusta al vocabulario ni al estilo específico de cada autor, sin embargo, cuando lo descongelé rindió mucho peor que cuando estaba congelado, ya que al tener que entender tantos datos desde 0 le costó y se ajustó muchisimo más lento incluso ajusté los parámetros, siento que tendría que haberle dado mínimo unas 40 épocas, en la versión final el primer resultado era epoch 1/10 | train loss 0.1749 acc 0.943 | val loss 0.0793 acc 0.976 y en la versión sin congelamiento el primero era epoch 1/25 | train loss 1.0553 acc 0.442 | val loss 1.0054 acc 0.534, la diferencia era notable.

## Metodología aplicada (Paso a paso)
1. EDA

Verifiqué la limpieza del dataset y analicé la distrución de clases, creando variables para medir la longitud de las oraciones. De aquí saqué varios datos relevantes, el largo de las oraciones ronda entre las 20-27 palabras para los 3 autores con Lovecraft (HPL) un poco por encima de los otros 2. Sin embargo al observar los valores atípicos, Mary Shelley (MWS) es quien produce las oraciones más largas de todo el dataset, llegando a superar las 800 palabras en casos extremos. Con esto pude pensar que la longitud de las oraciones también era una señal para diferenciar a los autores, no solo el vocabulario que usan.

2. Feature Engineering y selección de variables (Información mutua)
   
Apliqué mutual information para extraer las 25 palabras que aportan mayor información. Aquí pude ver que el top de la lista eran nombres propios de personajes especificos de las obras de los autores. Esto significa que el modelo podría no estar aprendiendo un "estilo de escritura" sino memorizando que nombres aparecen en qué obra (Si una oración menciona "Arkham" es lógico que sea de Lovecraft). Independiente de los nombres solo un segundo grupo de palabras (love, heart, misery, old, ancient) aportan una señal más relacionada con el tono y temática de los autores y textos. 

3. Entrenamiento y evaluación de Naive Bayes

Utilicé CountVectorizer (bag of words) para transformar los textos a números. Implementé el hiperparámetro mediante GridSearchCV usando validación cruzada. El modelo logró un accuracy cercano al 80%. En su matriz de confusión note que tiene una tasa de acierto muy alta, especialmente con Edgar Allan Poe (1248) y Mary Shelley (985)

4. Entrenamiento y evaluación de Bert

Para la red neuronal, congelé las capas base y entrené solo el clasificador final. Para evitar el overfitting que mencioné antes, configuré un Dropout del 50%, Weight Decay en el optimizador AdamW y un Early Stopping con paciencia de 3 épocas.
Al observar las curvas de entrenamiento, pude ver como la pérdida de validación bajó hasta estabilizarse, momento en el que el Early Stopping detuvo el proceso en la época 11 justo antes de que el modelo empezara a memorizar los datos. Al evaluar este modelo con los datos de prueba obtuvo un accuracy del 77.19% y un F1-macro de 77.07%. En su matriz de confusión, el modelo logro 1257 aciertos para Poe y 928 para Shelley.

5. Visualización del espacio latente (t-SNE)

Para entender como BERT estaba pensando, utilicé t-SNE para comprimir en 2 dimensiones las representaciones. En el gráfico se puede apreciar que BERT no separó de manera perfecta a los 3 autores pero si logró crear algunos clusters distingibles. Probablemente al estar congelado, lo que más pesó al principio eran los nombres de los personajes que Naive Bayes memorizó facilmente pero BERT logró agruparlos basándose en la estructura gramatical. 

##Conlusión y discusión crítica

Mi modelo de control clásico (Naive Bayes con 80%) superó levemente a mi modelo avanzado de Deep Learning (BERT con 77.19%).

Más que un error considero que tiene todo el sentido si reviso el análisis de información mutua. Naive Bayes ganó porque la técnica de bag of words le permitió enfocarse ciegamente en las palabras repetidas, memorizando los nombres de los personajes de cada obra.

Por otro lado, al usar BERT con Transfer Learning (congelado), el modelo venía pre-entrenado en inglés moderno de Wikipedia. No conocía el vocabulario gótico ni los nombres inventados del siglo XIX, y al no permitirle ajustar sus pesos internos no pudo darle la importancia necesaria a esas palabras claves. Sin embargo, el hecho de que BERT haya logrado un 77% de exactitud casi sin depender de los nombres es un gran logro porque desmuestra que la red neuronal fue capaz de clasificar a los autores aprendiendo a leer su estilo, ritmo y estructura gramatical de forma genuina. 

Intenté quitarle el congelado por una última vez para revisar si daba mejores resultados en algún punto que Naive Bayes, sin embargo tardaba demasiado y necesitaba muchas épocas. A pesar de que se equivocaba mucho más que en su versión final, aún así al probarlo era capaz de identificar correctamente a los autores. 



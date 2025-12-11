# Conclusiones — Ingeniería de Características y Discretización del Estado

---

## Ingeniería de Características del Estado

Para ambos agentes se definió una representación del estado basada en cuatro variables clave del entorno:

1. **Distancia al próximo tubo (`next_pipe_dist_to_player`)**  
2. **Distancia vertical entre el pájaro y el centro del hueco del tubo (`vertical_center_dist`)**  
3. **Velocidad vertical del pájaro (`player_vel`)**  
4. **Tamaño del hueco del tubo (`actual_gap`)**

Estas cuatro características capturan la información mínima necesaria para decidir si conviene saltar o continuar descendiendo.

---

## Discretización del Estado

Debido a que Flappy Bird es un entorno continuo, se aplicó discretización para permitir el entrenamiento mediante Q-Learning y para estandarizar la entrada del modelo neuronal.  
La discretización se realizó mediante bins uniformemente espaciados en cada dimensión:

- `x_bins`: 25 valores entre -150 y 150  
- `y_bins`: 70 valores entre -300 y 300  
- `v_bins`: 35 valores entre -20 y 20  
- `gap_bins`: 20 valores entre -200 y 200  

El estado final queda representado como una tupla de 4 valores discretos:  
`(distancia_discretizada, vertical_discretizada, velocidad_discretizada, gap_discretizado)`.

Este diseño permitió:
- Controlar la explosión del espacio de estados.  
- Dar al agente una representación estable del entorno.  
- Facilitar el entrenamiento del modelo neuronal mediante normalización posterior.

---

## Problemas Iniciales y Solución

Durante los primeros experimentos se observaron dos problemas principales:

### 1. **Discretización demasiado gruesa**  
La primera versión usaba muy pocos bins, lo que hacía que muchos estados distintos se mapearan al mismo valor.  
Consecuencias:
- El Q-Agent no podía aprender políticas estables.  
- El NN Agent no tenía suficientes diferencias en los inputs.  
- El agente tomaba decisiones erráticas y moría casi siempre.

---

# Conclusión General

La ingeniería de características correcta fue esencial para que ambos agentes pudieran aprender.  
El uso de:
- **Distancia al tubo**,  
- **Distancia vertical al centro del hueco**,  
- **Velocidad**,  
- **Tamaño del hueco**,  
combinado con discretización fina, produjo una representación confiable del entorno.

El Q-Agent sirvió para explorar el espacio y generar datos valiosos, pero la aproximación con red neuronal superó ampliamente sus limitaciones.

---
# Flappy Bird — Resultados del Entrenamiento del Agente Q-Table

Este documento resume los resultados del entrenamiento del agente basado en Q-learning para Flappy Bird, incluyendo estadísticas generales, gráficos y conclusiones.

## Estadísticas Generales

REWARD  
- Reward promedio: 0.259  
- Reward máximo: 152.0  
- Reward mínimo: -5.0  
- Reward en último episodio: -3.0  

MOVING AVERAGE (100 episodios)  
- MA promedio: 0.256  
- MA máximo: 17.620  
- MA último: 6.390  

EPSILON  
- Epsilon inicial: 0.999762  
- Epsilon final: 0.05  
- Epsilon mínimo observado: 0.05  

Q-TABLE  
- Estados únicos iniciales: 37  
- Estados únicos finales: 4783  
- Máximo tamaño de la Q-table: 4783  
- Crecimiento total: 4746 estados  

## Gráficos del Entrenamiento

### Tamaño de la Q-Table vs Episodios
![Tamaño Q-table vs Episodios](tamano%20q%20table%20vs%20episodios.png)

### Epsilon vs Episodio
![Epsilon vs Episodio](epsilon%20vs%20episodio.png)

### Recompensa por Episodio
![Recompensa por Episodio](recompesa%20por%20episodio.png)

### Reward Promedio Móvil (MA 100)
![Reward Promedio Móvil](reward%20promedio%20movil.png)

## Conclusiones del Agente Q-Table

El agente entrenado mediante Q-learning muestra un desempeño adecuado y evidencia aprendizaje progresivo. La presencia de episodios con recompensas altas indica que adquiere políticas efectivas, mientras que la tendencia ascendente del promedio móvil confirma una mejora estable del comportamiento.

El tamaño final de la Q-table, con cerca de 4800 estados, refleja una exploración amplia del espacio discretizado. Este crecimiento también evidencia una de las limitaciones principales del método: la expansión del número de estados cuando se utilizan discretizaciones más detalladas. Esto obliga a buscar un punto intermedio que permita capturar suficiente información del entorno sin que la tabla se vuelva inmanejable.

En términos generales, el agente funciona correctamente para este entorno, pero su rendimiento y capacidad de generalización están condicionados por las restricciones propias de las Q-tables. En escenarios más complejos o continuos.

---
# Flappy Bird — Resultados del Entrenamiento del NN Agent

## Estadísticas Generales

DESEMPEÑO DEL MODELO  
- El NN Agent generaliza mucho mejor que el Q-Agent.  
- Aprende una política estable y consistente.  
- Puede jugar durante largos períodos sin fallar.  
- Logra recompensas cercanas a **1000** (aprox. 1 hora de juego continuo).  
- Pierde muy pocas veces y se adapta a situaciones nuevas.  

PROBLEMAS DETECTADOS Y SOLUCIÓN  
- La discretización original era demasiado gruesa.  
- Muchos estados distintos terminaban representados como iguales.  
- Esto impedía que tanto la Q-table como el NN Agent aprendieran correctamente.  
- Tras refinar la discretización, el NN Agent pudo entrenar bien y generalizar correctamente.  

ESTADÍSTICAS DEL ENTRENAMIENTO  
(Ejemplo de convergencia observada en las primeras 60 épocas)  
- Pérdida inicial (loss): 0.75  
- Pérdida final (~época 180): 0.10  
- MAE inicial: 0.52  
- MAE final (~época 180): 0.15  
- val_loss bajó de 0.39 → 0.10  
- val_mae bajó de 0.38 → 0.17  

## Conclusiones del NN Agent

El NN Agent funciona mucho mejor que el Q-Agent debido a su capacidad para estimar valores Q en un espacio continuo de estados, sin depender de una tabla fija. Mientras que el agente basado en Q-learning estaba limitado por la discretización, el NN Agent pudo aprender representaciones más ricas del entorno.

Al inicio, el desempeño del agente neuronal era pobre, pero el problema no era el modelo, sino la discretización: faltaban estados y la representación era demasiado gruesa. Esto impedía que el agente tuviera información suficiente para aprender patrones útiles. Una vez corregida la discretización, el agente pudo entrenar correctamente.

El resultado es un modelo que:  
- Generaliza incluso a estados nunca vistos.  
- Aprende una política robusta y estable.  
- Puede jugar durante aproximadamente **una hora sin morir**, alcanzando premios cercanos a **1000**.  
- Muestra un comportamiento superior en consistencia y rendimiento frente al Q-Agent.

En resumen, el NN Agent demuestra que una buena ingeniería de estados combinada con un modelo funcional permite superar ampliamente las limitaciones del aprendizaje basado en Q-tables.
---
---
title: "Agente `paper-analyst` — Documentación y Guía de Uso"
author: "Prof. D.Sc. Aboud BARSEKH-ONJI  \nFacultad de Ingeniería — Universidad Anáhuac México"
date: "Primavera 2026"
geometry: "top=2.5cm, bottom=2.5cm, left=3cm, right=3cm"
fontsize: 11pt
lang: es
toc: true
toc-depth: 3
numbersections: true
colorlinks: true
linkcolor: "blue"
urlcolor: "blue"
header-includes: |
  \usepackage{booktabs}
  \usepackage{tcolorbox}
  \tcbuselibrary{skins}
  \usepackage{fancyhdr}
  \pagestyle{fancy}
  \fancyhf{}
  \rhead{\small Agente \texttt{paper-analyst}}
  \lhead{\small Redes Neuronales y SVM --- Anáhuac México}
  \rfoot{\small Página \thepage}
  \lfoot{\small Prof. D.Sc. Aboud BARSEKH-ONJI}
---

\newpage

# Introducción

## Propósito del Agente

El agente `paper-analyst` está diseñado para asistir a estudiantes de **último semestre especializados en Inteligencia Artificial** —en las carreras de Ingeniería, Actuaría o Negocios— durante la fase de revisión de literatura científica de su tesis o proyecto integrador.

Analizar un paper de IA/ML de forma rigurosa exige extraer contribuciones, evaluar la metodología estadística, juzgar la reproducibilidad y conectar los hallazgos con el trabajo propio. Sin guía, esta tarea consume horas y produce fichas incompletas. El agente estructura este proceso de manera sistemática y adaptada a la disciplina del alumno.

## ¿Por Qué Este Agente?

Los alumnos de último semestre con especialidad en IA comparten una necesidad transversal independientemente de su carrera:

| Carrera | Necesidad principal |
|---|---|
| Ingeniería | Comprender arquitecturas técnicas y evaluar su implementabilidad |
| Actuaría | Validar supuestos estadísticos y calibración probabilística |
| Negocios | Extraer valor de negocio, escalabilidad y consideraciones éticas |

Un agente genérico de chat responde preguntas sobre papers, pero no produce una **ficha de lectura estructurada**, no **compara múltiples papers en tabla sinóptica**, no **señala brechas explotables**, y no **ajusta su énfasis según la disciplina del alumno**. El agente `paper-analyst` hace todo esto de forma consistente.

## Diseño de Mínimo Privilegio

El agente tiene acceso solo a las herramientas estrictamente necesarias:

| Herramienta | Propósito | Por qué se incluye |
|---|---|---|
| `Read` | Leer PDFs y archivos de texto locales | El alumno puede cargar el paper como archivo |
| `Glob` | Localizar archivos en el proyecto | Encontrar PDFs en la carpeta de referencias |
| `Grep` | Buscar términos dentro de archivos largos | Localizar secciones específicas sin leer el paper completo |
| `WebFetch` | Descargar preprints de arXiv | Acceder a papers públicos sin que el alumno los descargue manualmente |

No se incluye `Bash` (riesgo de ejecución de comandos), `Write`/`Edit` (el agente no debe modificar nada) ni `WebSearch` (suficiente con `WebFetch` para URLs específicas de arXiv o Semantic Scholar).

---

# Instalación y Activación

## Paso 1: Crear el Directorio de Agentes

```bash
# Para disponibilidad solo en este proyecto:
mkdir -p .claude/agents/

# Para disponibilidad en todos los proyectos del usuario:
mkdir -p ~/.claude/agents/
```

## Paso 2: Copiar el Archivo de Definición

```bash
# Desde la carpeta Skill_and_agents/ del curso:
cp paper-analyst.md .claude/agents/

# O a nivel de usuario:
cp paper-analyst.md ~/.claude/agents/
```

## Paso 3: Verificar

El agente queda disponible inmediatamente. No se requiere reiniciar Claude Code. Para verificar, abrir Claude Code y preguntar: *"¿Qué agentes tienes disponibles?"* o simplemente pedirle que analice un paper.

---

# Guía de Uso

## Caso de Uso 1: Análisis de un Paper Local

El alumno descarga el PDF de un paper y lo coloca en su directorio de trabajo:

```
mi-tesis/
  referencias/
    attention_is_all_you_need_2017.pdf
    bert_devlin_2018.pdf
```

Luego le pide a Claude Code:

> *"Analiza el paper en referencias/attention_is_all_you_need_2017.pdf.
> Soy alumno de ingeniería y lo necesito para el marco teórico de mi
> tesis sobre procesamiento de lenguaje natural."*

### Lo que sucede internamente

1. Claude (orquestador) detecta la solicitud de análisis de paper
2. Selecciona el agente `paper-analyst` por sus triggers explícitos
3. Lanza el agente con el path del archivo y el contexto de disciplina
4. El agente usa `Read` para leer el PDF y `Grep` para localizar secciones
5. Genera la ficha completa siguiendo los 8 puntos del esquema estándar
6. El orquestador devuelve el análisis al alumno

### Salida Esperada (extracto)

---

**1. Identificación**

- **Título:** Attention Is All You Need (2017)
- **Autores:** Vaswani et al. — Google Brain / Google Research
- **Venue:** NeurIPS 2017 (Advances in Neural Information Processing Systems)
- **Ranking:** Top-1 conferencia en deep learning; más de 100,000 citas

**2. Contribución Principal**

Los autores proponen la arquitectura *Transformer*, que elimina la recurrencia y las convoluciones, basando el modelado de secuencias exclusivamente en mecanismos de atención (*self-attention*). Esto permite mayor paralelización durante el entrenamiento y establece el estado del arte en traducción automática (WMT 2014 EN-DE y EN-FR).

**3. Metodología**

- Arquitectura encoder-decoder con $N=6$ capas idénticas en cada componente
- Mecanismo de atención multi-cabeza: $\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,\ldots,\text{head}_h)W^O$
- Codificación posicional sinusoidal: $PE_{(pos,2i)} = \sin(pos/10000^{2i/d_{model}})$
- Entrenamiento con *label smoothing* ($\epsilon_{ls}=0.1$) y *warm-up* del optimizador Adam

**7. Conexión con tu Proyecto**

Puedes adoptar directamente el mecanismo de self-attention como capa de extracción de características en tu modelo de PLN. Si tu tesis propone una mejora a la eficiencia del Transformer (reducción de complejidad cuadrática), la Sección 3.2 de este paper es tu punto de partida obligatorio.

---

## Caso de Uso 2: Análisis Comparativo de Múltiples Papers

> *"Tengo tres papers en la carpeta referencias/. Compara sus metodologías
> de detección de anomalías con redes neuronales en una tabla."*

El agente lee los tres archivos, extrae las metodologías y produce una **tabla sinóptica comparativa** con dimensiones como: tipo de red, tipo de anomalía, dataset, métrica principal, F1-score reportado y disponibilidad de código.

## Caso de Uso 3: Evaluación de Reproducibilidad

> *"¿Qué tan reproducible es el paper bert_devlin_2018.pdf?
> ¿Podría replicarlo con los recursos de cómputo de la universidad?"*

El agente evalúa: disponibilidad del código, acceso a los datos de preentrenamiento, recursos computacionales requeridos (número de GPUs, tiempo de entrenamiento), y emite un veredicto de reproducibilidad en escala:

- **Alta:** código + datos públicos + entrenamiento en <24h en GPU simple
- **Media:** código disponible pero datos o cómputo limitantes
- **Baja:** sin código, datos privados o cómputo masivo requerido

## Caso de Uso 4: Conexión con el Proyecto Integrador

> *"Mi proyecto integrador usa Random Forest para clasificar riesgo de
> crédito. Analiza este paper de XGBoost y dime qué puedo incorporar
> en mi metodología."*

El agente (con perspectiva de actuaría activada) extrae las contribuciones del paper, evalúa si son aplicables al contexto del alumno, y sugiere experimentos adicionales (comparación de modelos, importancia de variables, calibración de probabilidades de default).

---

# Ejemplo Completo de Ficha de Lectura Generada

A continuación se presenta una ficha de lectura completa tal como la produce el agente, para el paper clásico de detección de anomalías con autoencoders.

---

**Ficha de Lectura — Generada por `paper-analyst`**

**1. Identificación**

- **Título:** Anomaly Detection using Autoencoders in High Performance Computing Systems
- **Autores:** Borghesi et al. — Università di Bologna
- **Venue:** AAAI-2019 Workshop on Artificial Intelligence for Data Center Operations
- **DOI:** 10.1609/aaai.v33i01.330 (aproximado)

**2. Contribución Principal**

Los autores aplican autoencoders con LSTM para detectar comportamientos anómalos en nodos de cómputo de alto rendimiento, sin necesidad de datos etiquetados. La clave es usar el *error de reconstrucción* como puntuación de anomalía, evitando la escasez de etiquetas en entornos industriales.

**3. Metodología**

- Autoencoder LSTM con arquitectura simétrica (encoder: 3 capas LSTM; decoder: 3 capas LSTM inversas)
- Entrenamiento solo con datos normales (*one-class learning*)
- Umbral de anomalía: media + $k\sigma$ del error de reconstrucción sobre datos de validación
- Ventana deslizante de 10 pasos temporales

**4. Datos y Experimentos**

- Dataset: telemetría de 516 nodos del clúster Marconi (CINECA, Italia)
- 30 días de operación normal para entrenamiento; 5 días con fallos conocidos para evaluación
- Sin acceso público al dataset (dato privado de infraestructura)

**5. Métricas y Resultados**

| Método | Precision | Recall | F1 |
|---|---|---|---|
| Autoencoder LSTM (propuesto) | 0.87 | 0.83 | 0.85 |
| PCA baseline | 0.71 | 0.65 | 0.68 |
| Isolation Forest | 0.74 | 0.70 | 0.72 |

**6. Análisis Crítico**

- *Fortalezas:* Enfoque no supervisado aplicable a dominios con pocas etiquetas; comparación honesta con tres baselines.
- *Debilidades:* Dataset privado impide reproducción exacta; umbral $k\sigma$ es sensible y se ajusta manualmente; no se reporta significancia estadística.
- *Reproducibilidad:* **Media** — el código no está publicado y los datos son privados, pero la arquitectura es lo suficientemente simple para replicar con PyTorch en datasets públicos (KPI-Anomaly, NAB).

**7. Conexión con tu Proyecto**

Si tu proyecto integrador propone detección de anomalías en datos financieros o industriales, puedes adoptar directamente la arquitectura LSTM-autoencoder. Sustituye el dataset por telemetría de tu dominio. El umbral $k\sigma$ es tu primer hiperparámetro a sintonizar.

**8. Palabras Clave para Búsqueda Relacionada**

`LSTM autoencoder anomaly detection`, `one-class learning time series`,
`reconstruction error threshold`, `unsupervised anomaly detection`,
`multivariate time series anomaly`, `deep autoencoder HPC`

---

# Consideraciones de Diseño

## Por Qué No Incluir `WebSearch`

El agente usa `WebFetch` (que requiere una URL exacta) en lugar de `WebSearch` (que realiza búsquedas abiertas). Esto es intencional:

1. **Foco:** el alumno debe traer el paper; el agente lo analiza. Evita que el agente se convierta en un buscador genérico.
2. **Reproducibilidad del análisis:** si el alumno da la URL de arXiv, el agente descarga exactamente ese paper, no una versión distinta.
3. **Seguridad:** `WebSearch` puede redirigir a sitios no deseados. `WebFetch` con URL explícita es predecible.

## Por Qué el Agente Responde en Español

El agente mantiene los **términos técnicos en inglés** (overfitting, embedding, benchmark, fine-tuning) porque así aparecen en la literatura y así los buscarán los alumnos en Google Scholar. El resto del análisis —incluyendo las secciones de crítica y conexión con el proyecto— está en español, que es el idioma de trabajo del alumno y de la institución.

## Ajuste por Disciplina

El campo `description` del frontmatter incluye "ingeniería, actuaría y negocios" explícitamente. Esto ayuda al orquestador a seleccionar este agente incluso cuando el alumno no menciona "paper" directamente, sino "artículo", "publicación" o "literatura".

---

# Actividad de Aprendizaje Propuesta

## Objetivo

Usar el agente `paper-analyst` para construir la sección de **Estado del Arte** de tu proyecto integrador o tesis.

## Instrucciones

1. Seleccionar 3 papers relevantes para tu tema de tesis (pueden ser de arXiv, Google Scholar o revistas del área).
2. Colocarlos en una carpeta `referencias/` dentro de tu directorio de proyecto.
3. Pedir al agente que analice cada uno por separado.
4. Pedir al agente que genere la **tabla comparativa sinóptica** de los tres papers.
5. Usar la sección "Conexión con tu Proyecto" de cada ficha para redactar los párrafos de tu Estado del Arte.

## Entregable

Un documento Word o LaTeX con el Estado del Arte redactado a partir de las fichas generadas, citando correctamente cada paper. El agente no escribe el Estado del Arte por ti — provee la materia prima estructurada para que tú lo redactes.

---

# Referencias

- Anthropic. (2025). *Claude Code — Documentación de Agentes*. <https://docs.anthropic.com/en/docs/claude-code/agents>
- Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS 2017*.
- Borghesi, A., et al. (2019). Anomaly detection using autoencoders in HPC systems. *AAAI Workshop*.
- Yao, S., et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models. *ICLR 2023*. <https://arxiv.org/abs/2210.03629>

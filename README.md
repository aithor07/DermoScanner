```markdown
# DermaScan AI: Sistema de Detección de Melanoma mediante Deep Learning

Este proyecto constituye el backend y el núcleo de inteligencia artificial para un sistema de triaje dermatológico. Utiliza redes neuronales convolucionales (CNN) con el objetivo de clasificar lesiones cutáneas en dos categorías: Benignas y Malignas (Melanoma).

El sistema ha sido implementado utilizando **Python**, **PyTorch** y **FastAPI**, aplicando técnicas avanzadas de Data Science para el balanceo de clases y Transfer Learning para la extracción de características.

## Estructura del Proyecto

El repositorio está organizado de la siguiente manera para separar la lógica de entrenamiento y la inferencia a través de la API:

```text
DermaScan-Backend/
├── data/                       # Dataset (PAD-UFES-20)
│   ├── images/                 # Imágenes crudas
│   ├── metadata.csv            # Metadatos clínicos y diagnósticos
│   └── train/val               # Estructura generada tras el pre-procesamiento
├── main.py                     # API REST (Backend) desarrollada con FastAPI
├── train.py                    # Script de entrenamiento, validación y Data Augmentation
├── modelo_melanoma.pth         # Pesos del modelo entrenado (Formato PyTorch estándar)
└── requirements.txt            # Dependencias del proyecto

```

## Fundamentos Técnicos y Metodología

### 1. Data Science y Pre-procesamiento (Pandas)

El dataset utilizado es el **PAD-UFES-20**, que contiene imágenes dermatoscópicas y clínicas. El procesamiento de datos se realizó mediante la librería **Pandas** para manipular el archivo `metadata.csv`.

* **Filtrado y Limpieza:** Se depuraron las entradas con diagnósticos ambiguos o imágenes corruptas.
* **Mapeo de Clases:** Se transformaron las etiquetas categóricas (ej: 'MEL', 'BCC', 'SCC' para malignos y 'ACK', 'NEV' para benignos) en una clasificación binaria numérica:
* `0`: Benigno
* `1`: Maligno


* **Gestión del Desbalance:** Se identificó un desbalance significativo entre muestras benignas y malignas, lo cual motivó la implementación de estrategias de ponderación en la función de pérdida para evitar sesgos.

### 2. Arquitectura del Modelo (Machine Learning)

Se optó por la técnica de **Transfer Learning** utilizando la arquitectura **MobileNetV2**.

* **Backbone:** Se utilizó MobileNetV2 pre-entrenada en ImageNet como extractor de características. Esta red ofrece un excelente equilibrio entre precisión y eficiencia computacional.
* **Classifier Head:** Se sustituyó la última capa lineal original (1000 clases) por una nueva capa totalmente conectada (`nn.Linear`) con:
* `in_features`: 1280
* `out_features`: 2 (Probabilidad Benigno vs Maligno)



### 3. Entrenamiento y Función de Pérdida Ponderada

El entrenamiento se ejecutó en `train.py` bajo las siguientes configuraciones críticas:

* **Weighted Cross Entropy Loss:** Para evitar que el modelo sesgara sus predicciones hacia la clase mayoritaria (Benigno), se implementó una función de pérdida ponderada. Se asignó un peso mayor a la clase "Maligno" (ratio aproximado 1:8). Esto penaliza severamente los Falsos Negativos, priorizando la sensibilidad del sistema (seguridad médica).
* **Optimizador:** Adam con un learning rate dinámico gestionado por un `StepLR Scheduler`, reduciendo la tasa de aprendizaje cada 10 épocas para afinar la convergencia.
* **Métricas:** Se monitorizó la precisión (Accuracy) global, priorizando la capacidad de detección de casos positivos en el conjunto de validación.

## API REST y Despliegue Local

El archivo `main.py` expone el modelo a través de una API REST construida con **FastAPI**, permitiendo realizar inferencias enviando imágenes mediante protocolo HTTP.

* **Endpoint `/predict`:** Recibe una imagen en formato binario.
* **Pre-procesamiento de Inferencia:** Aplica las mismas transformaciones que durante el entrenamiento (Resize a 224x224, conversión a Tensor y Normalización con medias y desviaciones estándar de ImageNet).
* **Respuesta:** Devuelve un objeto JSON con la clase predicha, las probabilidades calculadas mediante Softmax y mensajes de advertencia si la confianza es baja.

### Instrucciones de Ejecución

Para probar el sistema en un entorno local, siga los siguientes pasos:

1. **Instalar dependencias:**
Asegúrese de estar en el entorno virtual y ejecute:
```bash
pip install -r requirements.txt

```


2. **Iniciar el servidor:**
Utilice **Uvicorn** (servidor ASGI) para lanzar la aplicación. Ejecute el siguiente comando en la terminal:
```bash
uvicorn main:app --reload

```


* `main`: Hace referencia al archivo `main.py`.
* `app`: Es la instancia de FastAPI dentro del archivo.
* `--reload`: Habilita el reinicio automático si se detectan cambios en el código.


3. **Acceder a la Interfaz de Pruebas:**
FastAPI genera automáticamente documentación interactiva basada en el estándar OpenAPI. Abra su navegador y vaya a:
**`http://127.0.0.1:8000/docs`**
4. **Realizar una predicción:**
* Busque el endpoint **POST /predict**.
* Haga clic en "Try it out".
* Suba una imagen local en el campo "file".
* Haga clic en "Execute".
* El resultado aparecerá en la sección "Response body".



## Guía para la Captura de Imágenes

Debido a la alta sensibilidad del modelo (configurado para minimizar falsos negativos), la calidad de la imagen de entrada es crítica para obtener un diagnóstico fiable.

**Requisitos para una predicción correcta:**

1. **Macro/Primer Plano:** La lesión debe ocupar la parte central de la imagen. Evite fotos tomadas a más de 10-15 cm de distancia.
2. **Iluminación:** Use luz natural o flash. Las sombras oscuras pueden ser interpretadas erróneamente como pigmentación maligna.
3. **Enfoque:** La textura de la piel debe verse nítida. Si la imagen está borrosa, el modelo detectará bordes difusos, lo cual es un signo clínico de malignidad, provocando un Falso Positivo.
4. **Obstrucciones (Importante):** Evite la presencia de vello grueso sobre la lesión. El modelo puede confundir los pelos oscuros con el "retículo pigmentado atípico" (una estructura vascular asociada al melanoma). Si es posible, aparte el vello o tome la foto desde un ángulo limpio.

```

```

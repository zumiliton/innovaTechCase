# Arduino AI Assistant

Este repositorio contiene una aplicación de IA capaz de **identificar una placa Arduino a partir de una imagen** y, posteriormente, **responder preguntas sobre la placa identificada utilizando la documentación técnica proporcionada**.

La aplicación soporta tres tipos de placas:

* Arduino UNO
* Arduino MEGA
* Arduino NANO

La solución combina un modelo de visión para identificar la placa y un sistema RAG (Retrieval-Augmented Generation) para recuperar información relevante de la documentación y utilizarla como contexto para generar la respuesta.

La aplicación está dockerizada y permite elegir cómo ejecutar el **modelo de lenguaje utilizado en la segunda parte de la solución**. El sistema mantiene la misma funcionalidad de identificación y consulta de documentación independientemente de la opción seleccionada.

Se proporcionan dos configuraciones:

* **`local`**: el modelo de lenguaje se ejecuta localmente mediante `llama.cpp` y un modelo Qwen3.
* **`api`**: el modelo de lenguaje se ejecuta mediante un proveedor externo accesible a través de una API.

Estas configuraciones son **alternativas** y se seleccionan según las preferencias o los recursos disponibles para ejecutar la aplicación. La elección afecta únicamente a la ejecución del modelo de lenguaje; el sistema de recuperación de documentación y el resto de la aplicación permanecen iguales en ambas configuraciones.

La selección del perfil se realiza mediante el archivo `.env`. El repositorio incluye un `.env.example` sin credenciales como plantilla de configuración.

Este README contiene únicamente las instrucciones necesarias para **configurar y ejecutar la aplicación**. La arquitectura, decisiones técnicas, experimentos y resultados se documentan por separado en el informe del proyecto.


## Clonar el repositorio

En primer lugar, clona el repositorio en la máquina donde se vaya a ejecutar la aplicación:

```bash
git clone https://github.com/zumiliton/innovaTechCase.git
cd innovaTechCase
```

A partir de este directorio se ejecutarán todos los comandos indicados en las siguientes secciones del README.

> **Nota:** los modelos y otros archivos de gran tamaño no se almacenan en Git. Estos archivos deben descargarse o prepararse siguiendo las instrucciones correspondientes antes de iniciar la aplicación.


## Configuración del entorno

La configuración de la aplicación se realiza mediante un archivo `.env` situado en la raíz del repositorio.

El repositorio incluye un archivo `.env.example` como plantilla. Este archivo **no contiene credenciales ni tokens** y puede utilizarse como referencia para crear el `.env` local.

> **Importante:** el perfil de Docker Compose y el proveedor de LLM se seleccionan mediante las variables definidas en `.env`. Por tanto, antes de ejecutar la aplicación es necesario crear y configurar este archivo.

### Crear el archivo `.env`

Desde la raíz del proyecto:

```bash
cp .env.example .env
```

El archivo `.env` permite seleccionar entre dos perfiles de ejecución:

* `local`: ejecuta el LLM localmente mediante el contenedor `llama.cpp`.
* `api`: utiliza un LLM mediante una API externa.

### Perfil local

Para ejecutar la aplicación completamente en local:

```env
COMPOSE_PROFILES=local

LLM_PROVIDER=local
LLM_MODEL=

LLM_API_KEY=
LLM_BASE_URL=

LOG_LEVEL=INFO
```

En este modo no es necesario proporcionar ningún token de API.

### Perfil API

Para utilizar un proveedor externo:

```env
COMPOSE_PROFILES=api

LLM_PROVIDER=api
LLM_MODEL=openrouter/free

LLM_API_KEY=TU_API_KEY
LLM_BASE_URL=https://openrouter.ai/api/v1

LOG_LEVEL=INFO
```

En este caso es necesario introducir una API key válida en `LLM_API_KEY`.

La configuración incluida en `.env.example` deja deliberadamente este campo vacío para evitar publicar credenciales:

```env
LLM_API_KEY=
```

**No se deben incluir API keys, tokens ni otras credenciales en el repositorio.**

## Preparación de los modelos

Antes de iniciar la aplicación es necesario disponer de los modelos utilizados por el sistema.

Los modelos de gran tamaño **no están incluidos en el repositorio Git**. Para facilitar el despliegue, los **tres modelos utilizados por la aplicación están disponibles para su descarga desde la URL de Google Drive proporcionada junto con el proyecto**.

La estructura esperada es:

```text
models/
├── embeddings/
│   └── all-MiniLM-L6-v2/
├── llm/
│   └── Qwen3-8B-Q2_K.gguf
└── vision/
    └── resnet_tl/
        └── best.pt
```

### 1. Modelo de lenguaje local — Qwen3

Si se ha seleccionado el perfil `local`, **se debe utilizar el modelo Qwen3 cuantizado proporcionado para el proyecto**.

El archivo debe estar ubicado en:

```text
models/llm/Qwen3-8B-Q2_K.gguf
```

El modelo se ejecuta mediante `llama.cpp` dentro del contenedor correspondiente.

El modelo Qwen3 cuantizado puede descargarse directamente desde la **URL de Google Drive proporcionada junto con el proyecto**.

> Si se utiliza el perfil `api`, el modelo Qwen3 local no es necesario.

### 2. Modelo de embeddings — all-MiniLM-L6-v2

El sistema RAG utiliza `all-MiniLM-L6-v2` para generar los embeddings de las consultas y de la documentación.

El modelo debe estar disponible en:

```text
models/embeddings/all-MiniLM-L6-v2/
```

El modelo preparado para el proyecto puede descargarse directamente desde la **URL de Google Drive proporcionada junto con el proyecto**.

Este modelo es necesario tanto con el perfil `local` como con el perfil `api`, ya que forma parte del sistema de recuperación del RAG.

### 3. Modelo de visión — ResNet18

El clasificador de imágenes utiliza un modelo **ResNet18 ajustado para las tres clases de Arduino**.

Este modelo puede obtenerse de dos formas.

#### Opción A — Entrenar el modelo

El repositorio contiene el script utilizado para realizar el entrenamiento mediante transfer learning:

```text
vision_module/src/train/train_resnet18_tl
```

Este script permite generar el modelo de visión a partir del código y los datos correspondientes.

#### Opción B — Descargar el modelo ya entrenado

Para facilitar el despliegue, el modelo ResNet18 ya entrenado también está disponible en la **URL de Google Drive proporcionada junto con el proyecto**.

Una vez descargado, el checkpoint debe colocarse en:

```text
models/vision/resnet_tl/best.pt
```

Esta opción permite ejecutar directamente la aplicación sin volver a entrenar el modelo.

### Resumen

| Modelo           | ¿Necesario con `local`? | ¿Necesario con `api`? | Ubicación                             |
| ---------------- | ----------------------- | --------------------- | ------------------------------------- |
| ResNet18         | Sí                      | Sí                    | `models/vision/resnet_tl/best.pt`     |
| all-MiniLM-L6-v2 | Sí                      | Sí                    | `models/embeddings/all-MiniLM-L6-v2/` |
| Qwen3            | Sí                      | No                    | `models/llm/Qwen3-8B-Q2_K.gguf`       |

**Los tres modelos pueden descargarse desde la URL de Google Drive proporcionada junto con el proyecto**, lo que permite preparar rápidamente el entorno sin necesidad de entrenar o descargar individualmente cada componente.

La única excepción es el modelo de visión: si se desea reproducir el entrenamiento, también puede generarse mediante el script incluido en el repositorio.

## Ejecución con Docker Compose

Una vez configurado el archivo `.env` y preparados los modelos, la aplicación puede ejecutarse mediante Docker Compose.

### Primera ejecución

La primera vez que se ejecuta el proyecto es necesario construir las imágenes:

```bash
docker compose build
```

Una vez finalizado el proceso, se puede iniciar la aplicación con:

```bash
docker compose up
```

La aplicación quedará disponible en:

```text
http://localhost:8000
```

### Ejecuciones posteriores

Después de haber construido las imágenes, **no es necesario ejecutar `docker compose build` cada vez que se inicia la aplicación**.

Para iniciar la aplicación:

```bash
docker compose up
```

Para ejecutarla en segundo plano:

```bash
docker compose up -d
```

Para detener los contenedores:

```bash
docker compose down
```

### Cambiar entre `local` y `api`

La configuración del LLM se controla mediante el archivo `.env`.

Para cambiar de una configuración a otra:

1. Detener la ejecución actual:

```bash
docker compose down
```

2. Modificar `.env` y seleccionar el perfil deseado:

```env
COMPOSE_PROFILES=local
LLM_PROVIDER=local
```

o:

```env
COMPOSE_PROFILES=api
LLM_PROVIDER=api
```

3. Iniciar nuevamente la aplicación:

```bash
docker compose up
```

**No es necesario volver a ejecutar `docker compose build` al cambiar entre los perfiles `local` y `api`**, ya que ambos perfiles utilizan la misma aplicación y la selección del proveedor de LLM se realiza mediante las variables de entorno.

### ¿Cuándo es necesario volver a hacer `build`?

Se debe volver a construir la imagen cuando se produzcan cambios que afecten a la imagen Docker, por ejemplo:

* Cambios en el `Dockerfile`.
* Cambios en las dependencias instaladas durante el build.
* Cambios que requieran modificar la imagen base o su configuración.

En esos casos:

```bash
docker compose build
docker compose up
```

Los cambios realizados únicamente en `.env` **no requieren reconstruir la imagen**.

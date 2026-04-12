# Visión por Computadora 3
Este repositorio contiene la implementación del trabajo práctico final para la materia Visión por Computadora 3, correspondiente a la Maestría en Inteligencia Artificial dictada en FIUBA.

```
mia-vpc3/
├── notebooks/        # Exploración interactiva, pruebas de código
├── scripts/          # Scripts de automatización del pipeline de entrenamiento
├── runs/             # Resultados de las corridas
├── tests/            # Tests unitarios
│
├── pyproject.toml    # Configuración del proyecto (uv, pytest)
├── uv.lock           # Lock de versiones de dependencias
├── README.md         # Documentación simple
```

Para sincronizar dependencias, ejecutar
```bash
uv sync --dev
```

Se puede omitir el flag `--dev` si no se desean ejecutar los tests.

> [!IMPORTANT]
> Por defecto, los prompts de línea de comandos se ejecutan desde la raíz del repositorio.

## Experimentos

Para ejecutar un experimento se utiliza un archivo de configuración como el mostrado en [scripts/config_example.yaml](scripts/config_example.yaml).

```bash
uv run scripts/run_experiment.py <config_file_path>
```

El script incluye flags opcionales como `--use_colab` para cuando se ejecuta desde un entorno de Google Colab y `--test` para reducir a un tamaño mínimo el conjunto de datos a fin de testear rápidamente el correcto funcionamiento del pipeline.


## Suites

De forma similar a con los experimentos, para correr una suite de ellos se utiliza un script y un archivo de configuración como el mostrado en [scripts/suite_config_example.yaml](scripts/suite_config_example.yaml).

```bash
uv run scripts/run_suite.py <config_file_path>
```

El script de suite también incluye el flag opcional `--test` con el mismo efecto que en runs individuales.


## Pruebas

Para ejecutar las pruebas unitarias se debe correr la siguiente linea:

```bash
uv run pytest
```

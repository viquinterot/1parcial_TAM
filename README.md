# Comparación de Regresores sobre el Ames Housing Dataset

## Descripción
Este repositorio contiene el código y los recursos necesarios para comparar el desempeño de nueve regresores de **scikit-learn** sobre el [Ames Housing Dataset](https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset) utilizando:

- **Validación cruzada** de 5 folds  
- **Grid Search**, **Randomized Search** y **Optimización Bayesiana** (Optuna)  
- Métricas de desempeño: MAE, MSE, R² y MAPE  

El objetivo es identificar el modelo y la estrategia de ajuste de hiperparámetros que ofrezca el mejor compromiso entre sesgo y varianza en la predicción del precio de venta de viviendas en Ames, Iowa.

---

## Estructura del Repositorio

```
├── README.md
├── requirements.txt
├── AmesHousing.csv              # Datos originales (no versionados)
├── parcial_python.py            # Script principal de análisis (punto 2 del parcial)
├── ames_cache_colab/            # Carpeta de caché de resultados y tablas finales
│   └── df_final_results_colab.csv
└── streamlit_dashboard/         # Dashboard Streamlit (punto 3 del parcial)
    ├── 0_👋_Hello.py
    └── pages
        ├── 1_EDA.py
        └── 2_Model_Comparison.py
```

---

## Instalación

1. **Clonar el repositorio**  
   ```bash
   git clone https://github.com/tu-usuario/ames-housing-regression.git
   cd ames-housing-regression
   ```

2. **Crear un entorno virtual** (recomendado)  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Instalar dependencias**  
   ```bash
   pip install -r requirements.txt
   ```

---

## Datos

Descarga manualmente el conjunto de datos desde Kaggle y colócalo en la raíz del repositorio con el nombre `AmesHousing.csv`. El script `parcial_python.py` espera encontrarlo allí.

---

## Uso

### 1. Análisis y ajuste de modelos

Ejecuta el notebook convertido a script para:

- Preprocesar los datos (imputación, escalado, codificación ordinal y one-hot, agrupación de categorías raras, transformación logarítmica de la variable objetivo).  
- Definir pipelines y espacios de hiperparámetros para cada modelo.  
- Encontrar los mejores hiperparámetros con Grid Search, Randomized Search y Optuna.  
- Evaluar todos los modelos con validación cruzada externa de 5 folds y exportar resultados.

```bash
python parcial_python.py
```

Al finalizar, encontrarás en `ames_cache_colab/df_final_results_colab.csv` la tabla con:

| Model                  | TuningMethod      | MAE_log_mean | MSE_log_mean | R2_log_mean | MAPE_orig_mean | … |
|:-----------------------|:------------------|:------------:|:------------:|:-----------:|:--------------:|:--|
| RandomForestRegressor  | BayesOpt_Optuna   |     0.0245   |     0.0012   |    0.9123   |     7.56 %     | … |
| …                      | …                 |     …        |     …        |    …        |     …          | … |

---

### 2. Dashboard interactivo (Streamlit)

Para explorar los datos y comparar los tres mejores regresores:

```bash
cd streamlit_dashboard
streamlit run 0_👋_Hello.py
```

El dashboard incluye:

- **Página de Bienvenida**: Introducción al proyecto.  
- **EDA**: Visualizaciones interactivas de distribución, correlaciones y estadísticas descriptivas.  
- **Comparación de Modelos**: Tabla y gráficos de las métricas para los top-3 modelos según MSE logarítmico.

---

## Modelos y Hiperparámetros

| Modelo                          | Hiperparámetros clave                                      | Búsqueda                          |
|:--------------------------------|:---------------------------------------------------------:|:---------------------------------:|
| **LinearRegression**            | –                                                         | Baseline (sin ajuste)             |
| **Lasso**                       | α ∈ [5e-5, 1e-3]                                           | Grid / Random / BayesOpt          |
| **ElasticNet**                  | α ∈ [5e-4, 5e-3], l1_ratio ∈ [0.1, 0.9]                    | Grid / Random / BayesOpt          |
| **KernelRidge**                 | α ∈ [0.1, 1.0], γ ∈ [0.05, 0.5]                             | Grid / Random / BayesOpt          |
| **SGDRegressor**                | α ∈ [1e-5, 1e-3], penalty ∈ {l2, elasticnet}              | Grid / Random / BayesOpt          |
| **BayesianRidge**               | α_1, α_2, λ_1, λ_2 ∈ [1e-7, 1e-5]                          | Grid / Random / BayesOpt          |
| **GaussianProcessRegressor**    | α ∈ [0.1, 1.0]                                            | Grid / Random / BayesOpt          |
| **RandomForestRegressor**       | n_estimators ∈ [80, 300], max_depth ∈ [10, 50]             | Grid / Random / BayesOpt          |
| **SVR**                         | C ∈ [1, 50], γ ∈ [1e-3, 1.0], ε ∈ [0.05, 0.2]              | Grid / Random / BayesOpt          |

Cada estrategia de búsqueda está justificada en el script (rangos seleccionados según sensibilidad al overfitting y coste computacional) y utiliza como función objetivo la maximización del **neg_mean_squared_error**.

---

## Preprocesamiento

1. **Transformación de `SalePrice`**: log(1 + precio) para aproximación gaussiana.  
2. **Imputación**: mediana en numéricas, constante (‘Missing’ / ‘Missing_Nominal’) en categóricas.  
3. **Escalado**: RobustScaler en numéricas.  
4. **Codificación**:  
   - **OrdinalEncoder** en variables con orden natural (calidad, estado, exposición).  
   - **OneHotEncoder** (drop='first') en nominales.  
5. **Agrupación de categorías raras** (<1 %) a “Rare_Category” para evitar alta dimensionalidad.

---

## Métricas de Evaluación

- **MAE** (Error Absoluto Medio)  
- **MSE** (Error Cuadrático Medio)  
- **R²** (Coeficiente de Determinación)  
- **MAPE** (Error Porcentual Absoluto Medio)  

Se reportan medias y desviaciones estándar sobre los 5 folds de validación externa.

---

## Referencias

- Ames Housing Dataset (Kaggle)  
- [scikit-learn documentation](https://scikit-learn.org/)  
- [Optuna: A hyperparameter optimization framework](https://optuna.org/)  
- [Streamlit: Python framework para dashboards](https://streamlit.io/)

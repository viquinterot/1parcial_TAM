import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # Para visualización si es necesario (comentado por ahora)
import seaborn as sns # Para visualización si es necesario (comentado por ahora)

from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler, OrdinalEncoder # Usaremos RobustScaler para numéricas
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.base import clone

# Modelo GPR y Kernels
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

# TargetEncoder para variables categóricas nominales
try:
    from category_encoders import TargetEncoder
except ImportError:
    print("ERROR: category_encoders no está instalado. Por favor, instálalo: pip install category_encoders")
    exit()

# Optimización Bayesiana (usaremos Optuna aquí por su flexibilidad con pipelines complejos)
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    print("ADVERTENCIA: Optuna no está instalado. La Optimización Bayesiana no estará disponible.")
    print("Puedes instalarla con: pip install optuna")
    OPTUNA_AVAILABLE = False

# Utilidades
import time
import joblib
import os
import warnings
warnings.filterwarnings('ignore') # Ignorar warnings para una salida más limpia

# --- CONFIGURACIONES ---
MODEL_NAME_FOCUS = "GaussianProcessRegressor"
DATA_FILE = 'AmesHousing.csv'
CACHE_DIR_GPR = "gpr_cache" # Carpeta de caché específica para GPR

# --- 1. Carga y Limpieza Inicial de Datos ---
print("--- 1. Carga y Limpieza Inicial de Datos ---")
try:
    df_original = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"Error: Asegúrate de que '{DATA_FILE}' está en el mismo directorio.")
    exit()

df = df_original.copy()
print(f"Forma inicial del dataset: {df.shape}")

# Eliminar columnas con demasiados nulos o irrelevantes
cols_to_drop = ['Order', 'PID', 'Alley', 'Pool QC', 'Fence', 'Misc Feature']
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
print(f"Forma después de eliminar columnas iniciales: {df.shape}")

# Transformación Logarítmica de SalePrice
if 'SalePrice' in df.columns:
    df['SalePrice'] = np.log1p(df['SalePrice'])
    print("SalePrice transformado con log1p.")
else:
    print("ERROR: Columna 'SalePrice' no encontrada.")
    exit()

# --- 2. Definición de Características y Preprocesamiento ---
print("\n--- 2. Definición de Características y Preprocesamiento con Target Encoding ---")
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

# Identificar tipos de características
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features_all = X.select_dtypes(include='object').columns.tolist()

# Definición de mapeos ordinales (simplificado para este ejemplo, ajusta según tu EDA)
ordinal_mapping_quality = {'Missing': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5} # Asegúrate que 'Missing' sea una categoría
all_ordinal_mappings = { # Añade aquí todas tus columnas ordinales y sus mapeos
    'Exter Qual': ordinal_mapping_quality, 'Exter Cond': ordinal_mapping_quality,
    'Bsmt Qual': ordinal_mapping_quality, 'Bsmt Cond': ordinal_mapping_quality,
    'Heating QC': ordinal_mapping_quality, 'Kitchen Qual': ordinal_mapping_quality,
    'Fireplace Qu': ordinal_mapping_quality, 'Garage Qual': ordinal_mapping_quality,
    'Garage Cond': ordinal_mapping_quality,
    # Ejemplo de otras ordinales, DEBES COMPLETAR ESTO CON TUS MAPEADOS
    'Lot Shape': {'Missing': 0, 'IR3': 1, 'IR2': 2, 'IR1': 3, 'Reg': 4},
    'Land Slope': {'Missing': 0, 'Sev': 1, 'Mod': 2, 'Gtl': 3},
    'Bsmt Exposure': {'Missing': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4},
    'BsmtFin Type 1': {'Missing': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
    'BsmtFin Type 2': {'Missing': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
    'Functional': {'Missing':0, 'Sal':1, 'Sev':2, 'Maj2':3, 'Maj1':4, 'Mod':5, 'Min2':6, 'Min1':7, 'Typ':8},
    'Garage Finish': {'Missing': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3},
    'Paved Drive': {'Missing': 0, 'N': 1, 'P': 2, 'Y': 3},
    'Utilities': {'Missing': 0, 'ELO': 1, 'NoSeWa': 2, 'NoSewr': 3, 'AllPub': 4}, #Casi constante, podría eliminarse
}

ordinal_features_for_ct = [col for col in categorical_features_all if col in all_ordinal_mappings.keys() and col in X.columns]
nominal_features_for_ct = [col for col in categorical_features_all if col not in ordinal_features_for_ct and col in X.columns]

print(f"Numéricas: {len(numerical_features)}, Ordinales: {len(ordinal_features_for_ct)}, Nominales (para TargetEnc): {len(nominal_features_for_ct)}")

# Agrupación de categorías raras para nominales (antes de TargetEncoder)
for col in nominal_features_for_ct:
    if X[col].nunique() > 15:
        frequencies = X[col].value_counts(normalize=True)
        rare_categories = frequencies[frequencies < 0.005].index
        if len(rare_categories) > 0:
            # Para evitar SettingWithCopyWarning, es más seguro reasignar la columna
            # modificada en lugar de modificarla inplace con .loc si hay dudas sobre si X es una copia o una vista.
            # Crear una serie modificada y luego reasignarla a la columna del DataFrame.
            modified_column = X[col].replace(rare_categories, 'Rare_Nominal_Category')
            X[col] = modified_column # Esto generalmente evita el warning
            # Alternativamente, si X es definitivamente una copia, X.loc[:, col] = ... es seguro.
            # print(f"  En '{col}', se agruparon {len(rare_categories)} categorías como 'Rare_Nominal_Category'.") # (opcional)


# Preprocesadores
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])

ordinal_pipeline_steps = [('imputer_ord', SimpleImputer(strategy='constant', fill_value='Missing'))]
if ordinal_features_for_ct:
    ordinal_categories = [list(all_ordinal_mappings[col].keys()) for col in ordinal_features_for_ct]
    ordinal_pipeline_steps.append(
        ('ordinal_encoder', OrdinalEncoder(categories=ordinal_categories, handle_unknown='use_encoded_value', unknown_value=-1))
    )
ordinal_pipeline = Pipeline(steps=ordinal_pipeline_steps)


nominal_target_transformer = Pipeline(steps=[
    ('imputer_nom', SimpleImputer(strategy='constant', fill_value='UNK_Nominal')),
    ('target_encoder', TargetEncoder(handle_unknown='value', handle_missing='value', smoothing=10.0))
])

# ColumnTransformer
transformers_list_ct = []
if numerical_features:
    transformers_list_ct.append(('num', numerical_transformer, numerical_features))
if ordinal_features_for_ct: # Solo añadir si hay columnas ordinales
    transformers_list_ct.append(('ord', ordinal_pipeline, ordinal_features_for_ct))
if nominal_features_for_ct: # Solo añadir si hay columnas nominales
    transformers_list_ct.append(('nom_target', nominal_target_transformer, nominal_features_for_ct))

if not transformers_list_ct:
    print("ERROR: No hay transformadores válidos para ColumnTransformer. Revisa las listas de features.")
    exit()

preprocessor_gpr = ColumnTransformer(
    transformers=transformers_list_ct,
    remainder='drop',
    n_jobs=-1
)

# Probar el preprocesador para ver dimensionalidad
try:
    # fit_transform necesita X e y para TargetEncoder
    X_transformed_sample = preprocessor_gpr.fit_transform(X.head(5), y.head(5))
    print(f"Forma de X transformada (muestra): {X_transformed_sample.shape}")
    # Guardar el número de características transformadas para el kernel GPR si es necesario
    N_FEATURES_TRANSFORMED = X_transformed_sample.shape[1]
    if N_FEATURES_TRANSFORMED > 100 :
        print(f"ADVERTENCIA: Aún tienes {N_FEATURES_TRANSFORMED} características. GPR podría ser lento.")
    elif N_FEATURES_TRANSFORMED == 0:
        print("ERROR: El preprocesador no generó ninguna característica.")
        exit()
except Exception as e_preproc_test:
    print(f"Error al probar el preprocesador: {e_preproc_test}")
    exit()


# --- 3. Definición del Modelo GPR y Parámetros de Búsqueda ---
print("\n--- 3. Definición del Modelo GPR y Parámetros de Búsqueda ---")

# Nueva configuración del kernel con límites ajustados:
kernel_gpr_adjusted_bounds = C(constant_value=1.0, constant_value_bounds=(1e-4, 1e6)) * \
                             RBF(length_scale=1.0, length_scale_bounds=(1e-4, 1e4)) + \
                             WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-8, 1e1))

gpr_base_model = GaussianProcessRegressor(
    kernel=kernel_gpr_adjusted_bounds, # <--- USAR EL KERNEL CON LÍMITES AJUSTADOS
    random_state=42,
    normalize_y=True,
    n_restarts_optimizer=3 # Aumentado ligeramente
    # alpha se sigue tuneando externamente
)

pipeline_gpr = Pipeline(steps=[
    ('preprocessor', preprocessor_gpr),
    ('estimator', gpr_base_model)
])

gpr_params_config = {
    'params_grid': {
        'estimator__alpha': [1e-2, 0.1, 0.5, 1.0]
    },
    'params_random': {
        'estimator__alpha': np.logspace(-3, 0, 10).tolist()
    },
    'params_bayes_optuna': {
        'alpha': ('float', 1e-3, 1.5, False)
    }
}

# --- 4. Configuración de CV y Métricas ---
print("\n--- 4. Configuración de CV y Métricas ---")
CV_FOLDS_TUNING = 3
CV_FOLDS_EVALUATION = 5
kf_tuning = KFold(n_splits=CV_FOLDS_TUNING, shuffle=True, random_state=42)
kf_evaluation = KFold(n_splits=CV_FOLDS_EVALUATION, shuffle=True, random_state=123)
SCORING_TUNING = 'neg_mean_squared_error'

def mean_absolute_percentage_error_custom(y_true_orig, y_pred_orig):
    y_true_orig, y_pred_orig = np.array(y_true_orig), np.array(y_pred_orig)
    mask = y_true_orig != 0
    if not np.any(mask): return 0.0
    return np.mean(np.abs((y_true_orig[mask] - y_pred_orig[mask]) / y_true_orig[mask])) * 100

def get_gpr_cache_filename(search_type):
    os.makedirs(CACHE_DIR_GPR, exist_ok=True)
    return os.path.join(CACHE_DIR_GPR, f"{MODEL_NAME_FOCUS}_{search_type}.joblib")

def save_gpr_to_cache(data_to_save, search_type):
    filename = get_gpr_cache_filename(search_type)
    try:
        joblib.dump(data_to_save, filename)
        print(f"  Resultados para GPR ({search_type}) guardados en caché: {filename}")
    except Exception as e:
        print(f"  ERROR guardando caché para GPR ({search_type}): {e}")

def load_gpr_from_cache(search_type):
    filename = get_gpr_cache_filename(search_type)
    if os.path.exists(filename):
        try:
            print(f"  Cargando resultados para GPR ({search_type}) desde caché.")
            return joblib.load(filename)
        except Exception as e:
            print(f"  ERROR cargando caché {filename}: {e}. Se recomputará.")
            return None
    return None

# --- 5. Búsqueda de Hiperparámetros y Evaluación ---
print(f"\n--- 5. Búsqueda de Hiperparámetros y Evaluación para {MODEL_NAME_FOCUS} ---")
all_gpr_cv_results = []

# --- 5.1 GridSearchCV ---
search_method_gs = 'GridSearch'
print(f"\n--- {search_method_gs} para {MODEL_NAME_FOCUS} ---")
best_estimator_gs = None
cached_gs = load_gpr_from_cache(search_method_gs)
if cached_gs and 'best_estimator' in cached_gs:
    estimator_candidate_gs = cached_gs['best_estimator']
    if hasattr(estimator_candidate_gs, 'best_estimator_'):
        best_estimator_gs = estimator_candidate_gs.best_estimator_
    else: # Asumir que ya es un pipeline ajustado
        best_estimator_gs = estimator_candidate_gs
    print(f"  Mejores hiperparámetros (GridSearchCV) desde caché: {cached_gs.get('best_params', 'No guardados')}")
else:
    if gpr_params_config['params_grid']:
        start_time_gs = time.time()
        gs_cv = GridSearchCV(pipeline_gpr, gpr_params_config['params_grid'], cv=kf_tuning,
                             scoring=SCORING_TUNING, n_jobs=-1, verbose=1)
        try:
            gs_cv.fit(X, y)
            best_estimator_gs = gs_cv.best_estimator_
            print(f"    Mejores hiperparámetros (GridSearchCV): {gs_cv.best_params_}")
            save_gpr_to_cache({'best_estimator': gs_cv, 'best_params': gs_cv.best_params_}, search_method_gs) # Guardar el objeto de búsqueda
        except Exception as e:
            print(f"    Error en GridSearchCV para GPR: {e}")
        print(f"  GridSearchCV para GPR tomó {time.time() - start_time_gs:.2f} seg.")
    else:
        print("  No hay parámetros de grid para GPR. Ajustando modelo base.")
        pipeline_gpr.fit(X, y)
        best_estimator_gs = pipeline_gpr

# --- 5.2 RandomizedSearchCV ---
search_method_rs = 'RandomSearch'
print(f"\n--- {search_method_rs} para {MODEL_NAME_FOCUS} ---")
best_estimator_rs = None
N_ITER_RANDOM_GPR = 3
cached_rs = load_gpr_from_cache(search_method_rs)
if cached_rs and 'best_estimator' in cached_rs:
    estimator_candidate_rs = cached_rs['best_estimator']
    if hasattr(estimator_candidate_rs, 'best_estimator_'):
        best_estimator_rs = estimator_candidate_rs.best_estimator_
    else:
        best_estimator_rs = estimator_candidate_rs
    print(f"  Mejores hiperparámetros (RandomizedSearchCV) desde caché: {cached_rs.get('best_params', 'No guardados')}")
else:
    if gpr_params_config['params_random']:
        start_time_rs = time.time()
        rs_cv = RandomizedSearchCV(pipeline_gpr, gpr_params_config['params_random'], n_iter=N_ITER_RANDOM_GPR,
                                   cv=kf_tuning, scoring=SCORING_TUNING, n_jobs=-1,
                                   random_state=42, verbose=1)
        try:
            rs_cv.fit(X, y)
            best_estimator_rs = rs_cv.best_estimator_
            print(f"    Mejores hiperparámetros (RandomizedSearchCV): {rs_cv.best_params_}")
            save_gpr_to_cache({'best_estimator': rs_cv, 'best_params': rs_cv.best_params_}, search_method_rs)
        except Exception as e:
            print(f"    Error en RandomizedSearchCV para GPR: {e}")
        print(f"  RandomizedSearchCV para GPR tomó {time.time() - start_time_rs:.2f} seg.")
    else:
        print("  No hay parámetros de random para GPR.")
        if not best_estimator_gs :
            pipeline_gpr.fit(X, y)
            best_estimator_rs = pipeline_gpr

# --- 5.3 Optimización Bayesiana (Optuna) ---
search_method_bo = 'BayesOpt_Optuna'
print(f"\n--- {search_method_bo} para {MODEL_NAME_FOCUS} ---")
best_estimator_bo = None
N_TRIALS_OPTUNA_GPR = 3
if OPTUNA_AVAILABLE:
    cached_bo = load_gpr_from_cache(search_method_bo)
    if cached_bo and 'best_estimator' in cached_bo:
        best_estimator_bo = cached_bo['best_estimator']
        print(f"  Mejores hiperparámetros (Optuna) desde caché: {cached_bo.get('best_params_display', 'No guardados')}")
    else:
        if gpr_params_config['params_bayes_optuna']:
            start_time_bo = time.time()
            def objective_optuna_gpr(trial):
                params_to_set_optuna = {}
                alpha_config = gpr_params_config['params_bayes_optuna']['alpha']
                alpha_gpr = trial.suggest_float('alpha', alpha_config[1], alpha_config[2], log=alpha_config[3])
                params_to_set_optuna['estimator__alpha'] = alpha_gpr

                trial_pipeline_optuna = clone(pipeline_gpr)
                trial_pipeline_optuna.set_params(**params_to_set_optuna)
                try:
                    scores = cross_val_score(trial_pipeline_optuna, X, y, cv=kf_tuning, scoring=SCORING_TUNING, n_jobs=1)
                    return np.mean(scores)
                except Exception:
                    return -np.inf

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
            try:
                study.optimize(objective_optuna_gpr, n_trials=N_TRIALS_OPTUNA_GPR, show_progress_bar=True, n_jobs=1)
                best_params_optuna_raw = study.best_params
                final_pipeline_optuna = clone(pipeline_gpr)
                final_pipeline_optuna.set_params(**{f"estimator__{k}": v for k, v in best_params_optuna_raw.items()})
                final_pipeline_optuna.fit(X, y)
                best_estimator_bo = final_pipeline_optuna
                print(f"    Mejores hiperparámetros (Optuna): {best_params_optuna_raw}")
                save_gpr_to_cache({'best_estimator': best_estimator_bo, 'best_params_display': best_params_optuna_raw}, search_method_bo)
            except Exception as e:
                print(f"    Error en Optimización Bayesiana (Optuna) para GPR: {e}")
            print(f"  Optimización Bayesiana (Optuna) para GPR tomó {time.time() - start_time_bo:.2f} seg.")
        else:
            print("  No hay parámetros bayesianos para GPR.")
            if not best_estimator_gs and not best_estimator_rs:
                pipeline_gpr.fit(X,y)
                best_estimator_bo = pipeline_gpr
else:
    print("  Optuna no disponible. Omitiendo Optimización Bayesiana.")

# --- 5.4 Evaluación Final con CV de 5 Folds ---
estimators_to_evaluate = {
    "GridSearch": best_estimator_gs,
    "RandomSearch": best_estimator_rs,
    "BayesOpt_Optuna": best_estimator_bo
}

print(f"\n--- Evaluación Final (CV de {CV_FOLDS_EVALUATION} folds) para {MODEL_NAME_FOCUS} ---")
for tuning_method_name, final_estimator in estimators_to_evaluate.items():
    if final_estimator is None:
        print(f"  No hay mejor estimador de GPR para '{tuning_method_name}'. Omitiendo evaluación.")
        continue

    print(f"  Evaluando GPR (Mejor de '{tuning_method_name}')...")
    mae_scores_log, mse_scores_log, r2_scores_log, mape_scores_orig = [], [], [], []
    for train_idx, val_idx in kf_evaluation.split(X, y):
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val_log = y.iloc[train_idx], y.iloc[val_idx]
        current_fold_eval_pipeline = clone(final_estimator)
        try:
            current_fold_eval_pipeline.fit(X_fold_train, y_fold_train)
            y_pred_log = current_fold_eval_pipeline.predict(X_fold_val)
            mae_scores_log.append(mean_absolute_error(y_fold_val_log, y_pred_log))
            mse_scores_log.append(mean_squared_error(y_fold_val_log, y_pred_log))
            r2_scores_log.append(r2_score(y_fold_val_log, y_pred_log))
            y_fold_val_orig, y_pred_orig = np.expm1(y_fold_val_log), np.expm1(y_pred_log)
            mape_scores_orig.append(mean_absolute_percentage_error_custom(y_fold_val_orig, y_pred_orig))
        except Exception as e_cv_eval_fold:
            print(f"    ERROR en fold de CV evaluación para GPR ({tuning_method_name}): {e_cv_eval_fold}")
            mae_scores_log.append(np.nan); mse_scores_log.append(np.nan); r2_scores_log.append(np.nan); mape_scores_orig.append(np.nan)

    all_gpr_cv_results.append({
        'Model': MODEL_NAME_FOCUS, 'TuningMethod': tuning_method_name,
        'MAE_log_mean': np.nanmean(mae_scores_log), 'MAE_log_std': np.nanstd(mae_scores_log),
        'MSE_log_mean': np.nanmean(mse_scores_log), 'MSE_log_std': np.nanstd(mse_scores_log),
        'R2_log_mean': np.nanmean(r2_scores_log), 'R2_log_std': np.nanstd(r2_scores_log),
        'MAPE_orig_mean': np.nanmean(mape_scores_orig), 'MAPE_orig_std': np.nanstd(mape_scores_orig)
    })
    print(f"    Resultados Promedio CV GPR ({tuning_method_name}): MAE(log)={np.nanmean(mae_scores_log):.4f}, MSE(log)={np.nanmean(mse_scores_log):.4f}, R2(log)={np.nanmean(r2_scores_log):.4f}, MAPE(orig)={np.nanmean(mape_scores_orig):.2f}%")

# --- 6. Presentación de Resultados para GPR ---
print("\n\n--- 6. Presentación de Resultados Finales para GPR ---")
if all_gpr_cv_results:
    df_gpr_final_results = pd.DataFrame(all_gpr_cv_results)
    df_gpr_final_results = df_gpr_final_results.sort_values(by=['MSE_log_mean'])
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', '{:.4f}'.format)
    print("\nTabla de Resultados GPR (ordenada por mejor MSE_log_mean):")
    print(df_gpr_final_results)
else:
    print("No se generaron resultados para GPR.")

print("\n--- FIN DEL PROCESO PARA GPR ---")

# --- END OF FILE gp_adjusted_kernel.py ---
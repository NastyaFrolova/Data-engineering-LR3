from clearml import Task
from clearml.automation import HyperParameterOptimizer
from clearml.automation import DiscreteParameterRange
from clearml.automation.parameters import UniformParameterRange
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Сначала нужно запустить train.py
BASE_TASK_ID = '062b6861fe654655960c44cc60bc4c0d'

# Инициализация задачи-контроллера
controller_task = Task.init(
    project_name='Weather Forecast',
    task_name='HPO Controller (HyperParameterOptimizer)',
    task_type=Task.TaskTypes.optimizer,
    reuse_last_task_id=False
)


# Пространство поиска
hyper_parameters = [
    # Для float
    UniformParameterRange(name='hyperparams/learning_rate', min_value=0.01, max_value=0.3),
    UniformParameterRange(name='hyperparams/feature_fraction', min_value=0.6, max_value=1.0),

    # Для int
    DiscreteParameterRange(name='hyperparams/num_leaves', values=range(20, 101)),
    DiscreteParameterRange(name='hyperparams/n_estimators', values=range(50, 501)),
]



# Оптимизатор
hp_optimizer = HyperParameterOptimizer(
    base_task_id=BASE_TASK_ID,
    hyper_parameters=hyper_parameters,
    objective_metric_title="MAE",
    objective_metric_series="final_mae",
    objective_metric_sign="min",

    execution_queue="local",
    max_number_of_concurrent_tasks=2,
    total_max_jobs=10,
    time_limit_per_job=60,
)


print("Start HPO...")
hp_optimizer.start()


hp_optimizer.wait()
print("HPO Complete.")

top_exp = hp_optimizer.get_top_experiments(top_k=1)

best_task = top_exp[0]
print(f"Best ID: {best_task.id}")

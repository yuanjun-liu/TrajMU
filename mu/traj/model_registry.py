exec("import sys \nif __name__=='__main__':  sys.path.extend(['./'])")
from importlib import import_module
TASK_MODELS={
    'Sim': ['BLUE','TrajCL',],
    'Simp': ['MLSimp','S3',],
    'Map': ['TRMMA','DiffMM',],
    'Rec': ['TRMMA','ProDiff',],
}
TASK_DEFAULT_MODEL = {x:TASK_MODELS[x][0] for x in TASK_MODELS}
TASK_MODEL_CLASS = {
    ('Sim', 'BLUE'): ('mu.traj.model_blue', 'TrajSim'),
    ('Sim', 'TrajCL'): ('mu.traj.model_trajcl', 'TrajSimTrajCL'),
    ('Simp', 'MLSimp'): ('mu.traj.model_mlsimp', 'TrajSimp'),
    ('Simp', 'S3'): ('mu.traj.model_s3', 'TrajSimpS3'),
    ('Map', 'TRMMA'): ('mu.traj.model_trmma', 'TrajMap'),
    ('Map', 'DiffMM'): ('mu.traj.model_diffmm', 'TrajMapDiffMM'),
    ('Rec', 'TRMMA'): ('mu.traj.model_trmma', 'TrajRec'),
    ('Rec', 'ProDiff'): ('mu.traj.model_prodiff', 'TrajRecProDiff'),
}
TASK_MODEL_METRICS = {
    ('Sim', 'BLUE'): {
        'primary': 'MR',
        'secondary': 'HR10',
        'metric_keys': {'MR': 'MR_', 'MIA': 'MIA2', 'HR10': 'HR10_', 'HR5': 'HR5_', 'HR1': 'HR1_'},
    },
    ('Sim', 'TrajCL'): {
        'primary': 'MR','secondary': 'HR10',
        'metric_keys': {'MR': 'MR_', 'HR10': 'HR10_', 'HR5': 'HR5_', 'HR1': 'HR1_', 'MIA': 'MIA2'},
    },
    ('Simp', 'MLSimp'): {
        'primary': 'SED',
        'secondary': 'ITS',
        'metric_keys': {'SED': 'SEDwQ_', 'MIA': 'MIA3Q', 'F1': 'RangeF1', 'ITS': 'ITS_wQ_'},
    },
    ('Simp', 'S3'): {
        'primary': 'SED',
        'secondary': 'ITS',
        'metric_keys': {'SED': 'SED_', 'PED': 'PED_', 'DAD': 'DAD_', 'ITS': 'ITS_', 'MIA': 'MIA3'},
    },
    ('Map', 'TRMMA'): {
        'primary': 'F1',
        'secondary': 'Acc',
        'metric_keys': {'Acc': 'Acc_', 'MIA': 'MIA5', 'F1': 'F1_'},
    },
    ('Map', 'DiffMM'): {
        'primary': 'F1',
        'secondary': 'Acc',
        'metric_keys': {'Acc': 'Acc_', 'MIA': 'MIA5', 'F1': 'F1_'},
    },
    ('Rec', 'TRMMA'): {
        'primary': 'MAE',
        'secondary': 'Acc',
        'metric_keys': {'MAE': 'MAE_', 'MIA': 'MIA3', 'RMSE': 'RMSE_', 'Acc': 'Acc_', 'F1': 'F1_'},
    },
    ('Rec', 'ProDiff'): {
        'primary': 'MPPE','secondary': 'TC2000',
        'metric_keys': {
            'MTD': 'MTD_',
            'MPPE': 'MPPE_',
            'MAEPP': 'MAEPP_',
            'MAEPS': 'MAEPS_',
            'AVGTC': 'AVGTC_',
            'MaxTD': 'MaxTD_',
            'TC1000': 'TC1000_',
            'TC2000': 'TC2000_',
            'MIA': 'MIA3',
        },
    },
}
TASK_MODEL_TRAIN = {
    ('Sim', 'BLUE'): {
        'batch_size': 64,
        'train_epoch': 30,
    },
    ('Sim', 'TrajCL'): {
        'batch_size': 64,
        'train_epoch': 30,
    },
    ('Simp', 'MLSimp'): {
        'batch_size': 32,
        'train_epoch': 20,
    },
    ('Simp', 'S3'): {
        'batch_size': 20,
        'train_epoch': 20,
    },
    ('Map', 'TRMMA'): {
        'batch_size': 256,
        'train_epoch': 50,
    },
    ('Map', 'DiffMM'): {
        'batch_size': 32,
        'train_epoch': 30,
    },
    ('Rec', 'TRMMA'): {
        'batch_size': 256,
        'train_epoch': 50,
    },
    ('Rec', 'ProDiff'): {
        'batch_size': 256,
        'train_epoch': 100,
    },
}
def task_model_name(task, model_group=1):
    models=TASK_MODELS[task]
    assert 0<model_group<len(models)+1,'unsuporrt model index'
    return models[model_group-1]
def get_task_model_class(task, model_group=1):
    model = task_model_name(task, model_group)
    key = (task, model)
    if key not in TASK_MODEL_CLASS:
        raise KeyError(f'model class not registered: {task}-{model}')
    module_name, class_name = TASK_MODEL_CLASS[key]
    return getattr(import_module(module_name), class_name)
def is_default_task_model(task, model_group=1):
    return task_model_name(task, model_group) == TASK_DEFAULT_MODEL[task]
def task_result_name(task, model_group=1):
    model = task_model_name(task, model_group)
    if is_default_task_model(task, model_group):
        return task
    return f'{task}-{model}'
def get_metric_bundle(task, model_group=1):
    model = task_model_name(task, model_group)
    key = (task, model)
    if key not in TASK_MODEL_METRICS:
        raise KeyError(f'metric bundle not registered: {task}-{model}')
    return TASK_MODEL_METRICS[key]
def get_train_bundle(task, model_group=1):
    model = task_model_name(task, model_group)
    key = (task, model)
    if key not in TASK_MODEL_TRAIN:
        raise KeyError(f'train bundle not registered: {task}-{model}')
    return TASK_MODEL_TRAIN[key]

import os

os.environ['NEPTUNE_API_TOKEN']="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5NDc0ZmNhNi0wODFlLTRhYTktYjgwZS01MWJkMDMxNWJhNTAifQ=="
os.environ['NEPTUNE_PROJECT']="guilhermemg/icao-nets-training-2"
os.environ['NEPTUNE_NOTEBOOK_ID']="98a391a1-c710-40bd-aaf4-42c31862cbbe"
os.environ['NEPTUNE_NOTEBOOK_PATH']="training/exec_nas_experiment.ipynb"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # show only errors


import sys
if '..' not in sys.path:
    sys.path.insert(0, '..')


from src.base.experiment.evaluation.model_evaluator import DataSource
from src.exp_runner import ExperimentRunner

from exec_nas_v3_config import kwargs, APPROACH


runner = ExperimentRunner(**kwargs)

runner.load_training_data()
runner.sample_training_data()
runner.setup_data_generators()
runner.setup_experiment()

#runner.run_neural_architecture_search_v3()

best_arch = {'n_denses_0': 2, 'n_denses_1': 3, 'n_denses_2': 4, 'n_denses_3': 3, 'n_convs_0': 3, 'n_convs_1': 3, 'n_convs_2': 1, 'n_convs_3': 1}

runner.create_model(config=best_arch)
runner.visualize_model(outfile_path=f"training/figs/nas/nas_model_{APPROACH.name}.png")
runner.model_summary()
runner.train_model()
runner.draw_training_history()
runner.load_best_model()
runner.save_model()

runner.set_model_evaluator_data_src(DataSource.VALIDATION)
runner.test_model(verbose=False)


runner.set_model_evaluator_data_src(DataSource.TEST)
runner.test_model(verbose=False)

runner.finish_experiment()
from ray.tune.logger import LoggerCallback
from cnvrg import Experiment as CNVRGExperiment
import os

class CNVRGCallback(LoggerCallback):
    
    def __init__(self, tracked_metrics=None):
        self._cnvrg_metrics = tracked_metrics if tracked_metrics else []
        self._cnvrg_experiments = {}
        super(LoggerCallback, self).__init__()  
        
    def log_trial_start(self, trial):
        e = CNVRGExperiment.init()
        self._cnvrg_experiments[trial.trial_id] = e['slug']
        config = trial.config.copy()
        config.pop("callbacks", None)
        e.log_param("trial_id", trial.trial_id)
        e.log_param("run_id",trial.trial_id.split("_")[0])
        e.log(str(config))
        for item in config:
            e.log_param(item, config.get(item))
        e.log( "======")
        e.log(str(trial))
            
        
    def log_trial_result(self, iteration, trial, result):
        e = CNVRGExperiment(self._cnvrg_experiments[trial.trial_id])
        e.log(str(result))
        if self._cnvrg_metrics == []:
            self._cnvrg_metrics = [key for key in result]
            
        training_iteration = result['training_iteration']
        for key in self._cnvrg_metrics:
            try:
                value = float(result[key])
            except (ValueError, TypeError):
                continue
            e.log_metric(key, value, training_iteration)
    
    def log_trial_end(self, trial, failed):
        e = CNVRGExperiment(self._cnvrg_experiments[trial.trial_id])
        
        
        e.log("===== Logging Artifacts =====")
        from os import listdir
        files_list= [os.path.join(trial.logdir, p) for p in os.listdir(trial.logdir)]

        e.log_artifacts(files_list)
        e.finish(exit_status=int(failed))

 

 
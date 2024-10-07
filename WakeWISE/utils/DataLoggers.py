import os

import numpy as np

from WakeWISE.utils.logger import error, note

from datetime import datetime


class DataLogger:

    def __init__(self, target_dir = './SIM_LOGS'):

        if os.path.exists(target_dir) and not os.path.isdir(target_dir):
            error(f'Directory {target_dir} exists, but is not a folder.', "DataLoggers.py")
            raise Exception()
        elif not os.path.exists(target_dir):
            note(f'Directory {target_dir} does not exist, but will be created...', "DataLoggers.py")
            os.makedirs(target_dir)

        self.target_dir = target_dir



class NPLogger(DataLogger):

    def __init__(self, **kwargs) -> None:

        target_dir = kwargs.get('logger_dir', './SIM_LOGS')

        super().__init__(target_dir)

        self.data = []

    def __call__(self, wind_condition, turbines, results, rate, actions, op_modes) -> None:

        self.data.append(results[0, :, :])

    def end(self) -> None:

        # TODO: overwrite note
        self.data = np.stack(self.data, axis=0)
        np.save(f"{self.target_dir}/log_{datetime.now().strftime('%d%m%Y_%H%M%S')}.npy", self.data)
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from WakeWISE.utils.normalizer import Normalizer
from WakeWISE.WindTurbine import Turbine

base_path = Path(__file__).parent

class PowerSurrogate(nn.Module):

    def __init__(self):
        super(PowerSurrogate, self).__init__()

        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return x
    

class BaselinePowerSurrogate:
    """
    Experimental class for baseline removal using a surrogate model, only for power
    """

    MODEL_FILE_DICT = {
        'IEA 3.4 MW': 'BaselinePowerModels/IEA34MW.pth',
    }

    def __init__(self, turbine, **kwargs):

        if 'baseline_model_path' in kwargs:
            model_path = kwargs['baseline_model_path']
            assert os.path.exists(model_path), f'Error: Baseline model path {model_path} does not exist'
        else:
            assert isinstance(turbine, Turbine), 'Error: BaselinePowerSurrogate requires a Turbine object to determine the correct surrogate model'
            assert hasattr(turbine, 'turbine_type'), 'Error: Turbine object does not have a turbine_type attribute'
            assert turbine.turbine_type in self.MODEL_FILE_DICT, f'Error: No baseline model for turbine type {turbine.turbine_type}'
            model_path = os.path.join(base_path, self.MODEL_FILE_DICT[turbine.turbine_type])

        self.model = PowerSurrogate()
        self.model_data = torch.load(model_path)
        self.model.load_state_dict(self.model_data['state_dict'])
        self.model.eval()

        self.input_normalizer = Normalizer(self.model_data['Xmin'], self.model_data['Xmax'], type='min_max')
        self.output_normalizer = Normalizer(self.model_data['Ymin'], self.model_data['Ymax'], type='min_max')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model.to(self.device)

        self.turbine = turbine

    def run(self, wind_condition: dict) -> float:

        if wind_condition['ws'] < self.turbine.cutin or wind_condition['ws'] > self.turbine.cutout:
            return np.array([0.0])

        input = np.array([wind_condition['ws'], wind_condition['ti'], wind_condition['alpha'], 0.0])
        input = self.input_normalizer.normalize(input)
        input = torch.tensor(input, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            output = self.model(input)

        output = self.output_normalizer.denormalize(output.detach().cpu().numpy())
        return output
    
    def __call__(self, inputs):
        return self.run(inputs)
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data.data import Data

from WakeWISE.PropagationModels.PropagationModel import PropagationModel
from WakeWISE.PropagationModels.utils.Framework import WindFarmGNN
from WakeWISE.utils.logger import log
from WakeWISE.utils.WindFarmUtils.CoordinatesToGraph import to_graph
from WakeWISE.utils.WindFarmUtils.LayoutGenerator import layout_gen, matches_layout_pattern

base_path = Path(__file__).parent

class GNNSurrogate(PropagationModel):
    """
    Class for the surrogate model based on a GNN for the wind farm layout
    """

    def __init__(self, **kwargs) -> None:

        # Set the device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load some parameters
        self.kwargs = kwargs
        model_path = self.kwargs.get('model_path', f'{base_path}/utils/Models/best_22_07.pt') # Default model path is for the 3.4MW Model

        # Load the model
        assert os.path.exists(model_path), f'Error: File {model_path} does not exist'
        model_data = torch.load(model_path, map_location=self.device)
        self.model_dict, self.train_set_stats, self.config = model_data['model_state_dict'], model_data['trainset_stats'], model_data['config']

        # Initialize the model
        self.model = WindFarmGNN(**self.config.hyperparameters, **self.config.model_settings)
        self.model.trainset_stats = self.train_set_stats
        self.model.load_state_dict(self.model_dict)
        self.model.to(self.device)
        self.model.eval()

        self.set_layout()

        # Quick report to console
        num_t_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        log(f'Number of trainable parameters: {num_t_params:,}', origin = 'GNNSurrogate')

        # Initialize normalizers (old, not used anymore, but kept for reference)
        # norm_type = config.hyperparameters.norm_type
        # a_tag, b_tag = config.hyperparameters.norm_type.split('_')
        # self.node_feature_normalizer = Normalizer(train_set_stats['x'][a_tag], train_set_stats['x'][b_tag], type=norm_type)
        # self.global_feature_normalizer = Normalizer(train_set_stats['globals'][a_tag], train_set_stats['globals'][b_tag], type=norm_type)
        # self.edge_feature_normalizer = Normalizer(train_set_stats['edges'][a_tag], train_set_stats['edges'][b_tag], type=norm_type)
        # self.output_normalizer = Normalizer(train_set_stats['y'][a_tag], train_set_stats['y'][b_tag], type=norm_type)

        self.node_dim = self.config.hyperparameters.node_feature_dim
        self.edge_dim = self.config.hyperparameters.edge_feature_dim
        self.n_outputs = self.train_set_stats['y']['mean'].shape[0]

    def set_layout(self, layout = None, layout_D = None) -> None:

        if layout is None:
            layout = self.kwargs.get('layout', f'{base_path}/../data/WindFarmLayouts/Lillgrund_FC_shifted.pt')
        if layout_D is None:
            layout_D = self.kwargs.get('layout_D', 500)

        layout_D = np.random.uniform(layout_D[0], layout_D[1]) if isinstance(layout_D, list) else layout_D

        # 5 options for layout:
        # 1: pattern code (e.g. s4 for 4x4 square grid)
        # 2: path to .pt file
        # 3: path to .csv file
        # 4: torch_geometric.data.Data object
        # 5: list of coordinates
        # 6: int for nxn grid of turbines

        # Case 1, 2 and 3: Load layout from file or string
        if isinstance(layout, str):

            if layout == 'random':
                raise NotImplementedError('Random layout not implemented yet; use \'rn\' instead, where n is the size of the grid')

            elif matches_layout_pattern(layout):

                log(f'Generating layout from pattern {layout} with {layout_D}m spacing', origin = 'GNNSurrogate')
                layout = layout_gen(layout, D=layout_D)
                self.graph_layout = to_graph(layout, connectivity=self.kwargs.get('connectivity', 'fully_connected'))

            else:

                log('Loading layout from file', origin = 'GNNSurrogate')
                
                # Check if file exists
                assert os.path.exists(layout), f'Error: File {layout} does not exist'

                # If file is a .pt file, load it using torch
                if layout.endswith('.pt'):

                    log(f'Loading layout from {layout} file', origin = 'GNNSurrogate')
                    self.graph_layout = torch.load(layout)

                # If file is a .csv file, load it using pandas
                elif layout.endswith('.csv'):

                    import pandas as pd
                    log(f'Loading layout from {layout} file', origin = 'GNNSurrogate')

                    # Load the csv file using pandas
                    coordinates_df = pd.read_csv(layout)
                    coordinates_array = coordinates_df[['X', 'Y']].to_numpy()

                    # Convert the coordinates to a graph
                    self.graph_layout = to_graph(coordinates_array, connectivity=self.kwargs.get('connectivity', 'fully_connected'))
                
        # Case 4: Load layout from Data object
        elif isinstance(layout, Data):

            log('Loading layout from Data object', origin = 'GNNSurrogate')
            self.graph_layout = layout

        # Case 5: Load layout from list of coordinates
        elif isinstance(layout, np.ndarray):
            
            log('Loading layout from numpy array', origin = 'GNNSurrogate')
            self.graph_layout = to_graph(layout, connectivity=self.kwargs.get('connectivity', 'fully_connected'))

        # Case 6: Load layout from integer for nxn grid
        elif isinstance(layout, int):

            log(f'Loading layout by creating {layout}x{layout} grid with {layout_D}m spacing', origin = 'GNNSurrogate')
            h = layout * layout_D / 2
            layout = np.array([[i*layout_D-h, j*layout_D-h] for i in range(layout) for j in range(layout)])
            self.graph_layout = to_graph(layout, connectivity=self.kwargs.get('connectivity', 'fully_connected'))

        else:
            raise ValueError('Invalid layout input')

        self.n_turbines = self.graph_layout.num_nodes
        self.num_edges = self.graph_layout.num_edges
        self.graph_layout = self.graph_layout.to(self.device)
        self.graph_layout.edge_attr = torch.clip(self.graph_layout.edge_attr, self.train_set_stats['edges']['min'][:2], None)
        self.graph_layout.edge_attr_original = self.graph_layout.edge_attr.clone().detach()

    def __len__(self) -> int:

        return self.n_turbines
    
    def get_positions(self) -> torch.Tensor:

        return self.graph_layout.pos.clone().detach().cpu()
    
    def __call__(self, wind_condition: np.ndarray, turbine_states: np.ndarray) -> np.ndarray:
        return self.run(turbine_states)
    
    def precompute_edges(self, wind_condition: np.ndarray) -> None:

        self.graph_layout.edge_attr = self.graph_layout.edge_attr_original.clone()
        self.graph_layout.globals = torch.tensor(wind_condition, device=self.device).float().unsqueeze(0)

        # Add relative wind direction to edge attributes
        if self.config.hyperparameters.rel_wd is True:
            edge_rel_wd = math.radians(self.graph_layout.globals[0, 1]) - self.graph_layout.edge_attr[:, 1]
            self.graph_layout.edge_attr = torch.cat((self.graph_layout.edge_attr, edge_rel_wd.unsqueeze(1)), dim=1)

    def run(self, turbine_states: np.ndarray) -> np.ndarray:
        """
        Run the surrogate model to get the power output, local ws and ti and the DEL values for the wind farm layout
        
        Args:
            wind_condition (np.ndarray): the wind condition for the wind farm
            turbine_states (np.ndarray): the states of the turbines in the wind farm

        Returns:
            np.ndarray: the power output, local ws and ti and the DEL values for the wind farm layout
        """

        # Construct the graph layout
        self.graph_layout.x = torch.tensor(turbine_states, device=self.device).float()

        with torch.no_grad():
            
            # Unused, kept for reference
            # self.graph_layout.globals = self.global_feature_normalizer.normalize(self.graph_layout.globals)
            # self.graph_layout.edge_attr = self.edge_feature_normalizer.normalize(self.graph_layout.edge_attr)
            # self.graph_layout.x = self.node_feature_normalizer.normalize(self.graph_layout.x)

            data = self.model(self.graph_layout)

            # By doing this, already in form of each row corresponds to a turbine, and has row of power, ws, ti, dels
            # data = self.output_normalizer.denormalize(data.x).cpu().numpy() # Unused, kept for reference
            data = data.x.cpu().numpy()
            data = np.maximum(data, 0) # Clip negative values to 0. Negative values make no sense

        # Order: power, local ws, local ti, DEL Flapwise, DEL Edgewise, DEL TTYaw, DEL TBSS, DEL TBFA
        # data.shape = (n_turbines, 8)
        return data.astype(np.float64)
    
    def run_with_grad(self, turbine_states: torch.Tensor) -> torch.Tensor:

        # Experimental function for gradient descent with autograd
        self.model.requires_grad_(False)
        self.graph_layout.edge_attr.requires_grad_(False)
        self.graph_layout.globals.requires_grad_(False)
        self.graph_layout.x = turbine_states.to(self.device)

        data = self.model(self.graph_layout).x

        data = torch.clip(data, 0, None)

        return data
from abc import ABC, abstractmethod

class MaintenanceModel(ABC):

    def __init__(self, g, **kwargs) -> None:
        """
        Initializes the maintenance model
        
        Args:
        """

        self.maintaining = None
        self.g = g
        self.sf = kwargs.get('damage_safety_factor', 1.00)
        self.failure_limit = 1 / self.sf
        self.interventions = 0

    @abstractmethod
    def run(self, turbines: list, undo: bool = False, **conditions):
        "Runs the maintenance model for the given turbines"

    @abstractmethod
    def reset(self):
        "Resets the maintenance model"
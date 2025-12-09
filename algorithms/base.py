from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    def select_action(self, obs, deterministic=False):
        pass
    
    @abstractmethod
    def train_step(self):
        pass
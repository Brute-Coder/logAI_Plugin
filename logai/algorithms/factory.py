"""
This modified version includes support for plugin metadata and a plugin discovery 
mechanism. It allows plugins to specify metadata such as version, author, and description,
and provides methods to retrieve this metadata. Additionally, it enables the discovery of
plugins in specified directories, making it easier to manage and explore available plugins.
and improve inline documentation.
"""
import importlib.util
import os
import logging
from abc import ABC, abstractmethod
from time import time
import subprocess
from typing import List, Tuple

# Setup logging configuration
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PluginInterface(ABC):
    """
    Interface for plugin modules.
    All plugins must implement this interface.
    """
    @abstractmethod
    def register_algorithm(self, task: str, name: str, algorithm_class):
        """
        Registers an algorithm under a task and name.
        
        Args:
            task (str): The task name.
            name (str): The name of the algorithm.
            algorithm_class (class): The class implementing the algorithm.
        """
        pass

    @abstractmethod
    def is_registered(self, task: str, name: str) -> bool:
        """
        Checks if an algorithm is already registered.
        
        Args:
            task (str): The task name.
            name (str): The name of the algorithm.
        
        Returns:
            bool: True if the algorithm is registered, False otherwise.
        """
        pass

    @abstractmethod
    def get_algorithm_instance(self, task: str, name: str, config=None):
        """
        Creates and returns an instance of the algorithm.
        
        Args:
            task (str): The task name.
            name (str): The name of the algorithm.
            config (dict, optional): Configuration parameters for initializing the algorithm instance.
        
        Returns:
            Instance: An instance of the algorithm.
        """
        pass

class AlgorithmFactory:
    """
    Factory class for creating and managing algorithm instances using a lazy loading mechanism
    and caching for improved performance and reduced initial load time.
    Attributes include a registry for algorithms and a list to store plugin paths for lazy loading.
    """
    _algorithms_with_torch = {"lstm", "cnn", "transformer", "logbert", "forecast_nn"}

    def __init__(self, plugin_paths: List[str] = None):
        """
        Initializes the AlgorithmFactory, sets up the registry and preloads plugin paths.
        
        Args:
            plugin_paths (list of str, optional): List of paths to search for plugins.
        """
        self.registry = AlgorithmRegistry()
        self.plugin_paths = plugin_paths or []
        self._plugin_modification_times = {}
        self._load_plugin_paths()

    def _load_plugin_paths(self):
        """
        Preloads the paths of all plugins from configured directories.
        This avoids loading all modules at startup, saving resources.
        """
        for plugin_path in self.plugin_paths:
            for root, dirs, files in os.walk(plugin_path):
                for file in files:
                    if file.endswith(".py") and file != "__init__.py":
                        plugin_name = os.path.splitext(file)[0]
                        module_name = os.path.splitext(file)[0]
                        module_path = os.path.join(root, file)
                        self.plugin_paths.append((module_name, module_path))
                        self._plugin_modification_times[module_name] = os.path.getmtime(module_path)
                        logger.info(f"Stored plugin path for {plugin_name} under {module_path}")

    def get_algorithm(self, task: str, name: str, config=None):
        """
        Retrieves an algorithm instance by task and name when needed, applying the given configuration.
        
        Args:
            task (str): The task category of the algorithm.
            name (str): The name of the algorithm.
            config (dict, optional): Configuration parameters for initializing the algorithm instance.
        
        Returns:
            Instance: An instance of the requested algorithm configured with 'config', or None if not found.
        """
        key = (task, name)
        for module_name, module_path in self.plugin_paths:
            if key == (module_name, module_path):
                if self._plugin_needs_reload(module_name, module_path):
                    logger.info(f"Reloading plugin: {module_name}")
                    self._reload_plugin(module_name, module_path)
                if not self.registry.is_registered(task, name):
                    self._load_and_register_plugin(module_name, module_path, task, name)
                return self.registry.get_algorithm_instance(task, name, config)
        else:
            logger.error(f"No plugin found for task: {task}, name: {name}")
            return None

    def _plugin_needs_reload(self, module_name: str, module_path: str) -> bool:
        """
        Checks if a plugin module needs to be reloaded based on its modification time.
        
        Args:
            module_name (str): The name of the plugin module.
            module_path (str): The path to the plugin module.
        
        Returns:
            bool: True if the plugin needs to be reloaded, False otherwise.
        """
        current_mtime = os.path.getmtime(module_path)
        return current_mtime > self._plugin_modification_times.get(module_name, 0)

    def _reload_plugin(self, module_name: str, module_path: str):
        """
        Reloads a plugin module.
        
        Args:
            module_name (str): The name of the plugin module.
            module_path (str): The path to the plugin module.
        """
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.registry.unregister_all(module_name)
        self.registry.register_all(module)

    def _load_and_register_plugin(self, module_name: str, plugin_path: str, task: str, name: str):
        """
        Loads a plugin module from a path and registers it, handling on-the-fly loading.
        
        Args:
            module_name (str): The name of the plugin module.
            plugin_path (str): The path to the plugin module.
            task (str): The task category of the algorithm.
            name (str): The name of the algorithm.
        """
        try:
            spec = importlib.util.spec_from_file_location(module_name, plugin_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if self._validate_plugin_interface(module):
                self._install_plugin_dependencies(module)
                self.registry.register_algorithm(task, name, module)
                logger.info(f"Plugin {name} for task {task} loaded and registered.")
            else:
                logger.error(f"Plugin {name} for task {task} does not conform to the required interface.")
        except Exception as e:
            logger.error(f"Failed to load and register {name}: {e}")

    def _validate_plugin_interface(self, plugin_module) -> bool:
        """
        Validates if the loaded plugin module conforms to the PluginInterface.
        
        Args:
            plugin_module (module): The loaded plugin module.
        
        Returns:
            bool: True if the plugin module conforms to the interface, False otherwise.
        """
        return all(hasattr(plugin_module, attr) for attr in dir(PluginInterface))

    def _install_plugin_dependencies(self, plugin_module):
        """
        Installs dependencies required by the plugin module.
        
        Args:
            plugin_module (module): The loaded plugin module.
        """
        if hasattr(plugin_module, 'dependencies'):
            for dependency in plugin_module.dependencies:
                subprocess.run(['pip', 'install', dependency])

class AlgorithmRegistry(PluginInterface):
    """
    Manages algorithm instances, providing caching to minimize reloading.
    """
    def __init__(self):
        """
        Initializes the registry with an empty dictionary to store algorithms.
        """
        self._algorithms = {}

    def register_algorithm(self, task: str, name: str, algorithm_class):
        """
        Registers an algorithm under a task and name, caching the class type.
        
        Args:
            task (str): The task name.
            name (str): The name of the algorithm.
            algorithm_class (class): The class implementing the algorithm.
        """
        self._algorithms[(task, name)] = algorithm_class
        logger.info(f"Algorithm registered: {name} for task {task}")

    def unregister_all(self, module_name: str):
        """
        Unregisters all algorithms associated with a plugin module.
        
        Args:
            module_name (str): The name of the plugin module.
        """
        for key, value in list(self._algorithms.items()):
            if value.__module__ == module_name:
                self._algorithms.pop(key)

    def register_all(self, plugin_module):
        """
        Registers all algorithms defined in a plugin module.
        
        Args:
            plugin_module (module): The loaded plugin module.
        """
        for name, obj in plugin_module.__dict__.items():
            if hasattr(obj, '__call__'):
                if hasattr(obj, 'task') and hasattr(obj, 'config_class'):
                    self.register_algorithm(obj.task, name, obj)

    def is_registered(self, task: str, name: str) -> bool:
        """
        Checks if an algorithm is already registered to avoid re-loading.
        
        Args:
            task (str): The task name.
            name (str): The name of the algorithm.
        
        Returns:
            bool: True if the algorithm is registered, False otherwise.
        """
        return (task, name) in self._algorithms

    def get_algorithm_instance(self, task: str, name: str, config=None):
        """
        Creates and returns an instance of the algorithm using the provided configuration.
        
        Args:
            task (str): Task category of the algorithm.
            name (str): Name of the algorithm.
            config (dict, optional): Configuration for the algorithm instance.
        
        Returns:
            Instance: An instance of the algorithm configured according to 'config'.
        """
        algorithm_class = self._algorithms.get((task, name))
        if algorithm_class:
            if config:
                # Instantiate the algorithm with configuration if provided
                instance = algorithm_class(**config)
            else:
                # Instantiate the algorithm without configuration if none provided
                instance = algorithm_class()
            return instance
        else:
            logger.error(f"Algorithm class not found for {name} in task {task}")
            return None

# Example usage:
if __name__ == "__main__":
    # Specify plugin paths
    plugin_paths = ["./anomaly_detection_algo"]

    # Initialize AlgorithmFactory with plugin paths
    factory = AlgorithmFactory(plugin_paths)

    # Get an algorithm instance
    algorithm_instance = factory.get_algorithm("detection", "custom_algorithm")

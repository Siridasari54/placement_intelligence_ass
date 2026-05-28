"""Dependency Injection Container for managing service dependencies."""

from typing import Dict, Any, Type, TypeVar, Optional, Callable, Tuple
from abc import ABC, abstractmethod
import inspect
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class IServiceProvider(ABC):
    """Interface for service provider."""
    
    @abstractmethod
    def get_service(self, service_type: Type[T]) -> T:
        """Get a service instance.
        
        Args:
            service_type: Type of service to retrieve
            
        Returns:
            Service instance
        """
        pass


class ServiceContainer(IServiceProvider):
    """Dependency injection container for managing service lifecycle."""
    
    def __init__(self):
        """Initialize the service container."""
        self._services: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable] = {}
        self._singletons: Dict[Type, Any] = {}
        self._config: Dict[str, Any] = {}
        
        logger.info("ServiceContainer initialized")
    
    def register_singleton(self, service_type: Type[T], instance: T) -> None:
        """Register a singleton service instance.
        
        Args:
            service_type: Type of service
            instance: Service instance
        """
        self._singletons[service_type] = instance
        logger.info(f"Registered singleton: {service_type.__name__}")
    
    def register_factory(self, service_type: Type[T], factory: Callable[..., T]) -> None:
        """Register a factory function for creating services.
        
        Args:
            service_type: Type of service
            factory: Factory function
        """
        self._factories[service_type] = factory
        logger.info(f"Registered factory: {service_type.__name__}")
    
    def register_transient(self, service_type: Type[T], factory: Callable[..., T]) -> None:
        """Register a transient service (new instance each time).
        
        Args:
            service_type: Type of service
            factory: Factory function
        """
        self._factories[service_type] = factory
        logger.info(f"Registered transient: {service_type.__name__}")
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """Set configuration for the container.
        
        Args:
            config: Configuration dictionary
        """
        self._config = config
        logger.info("Container configuration updated")
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self._config.get(key, default)
    
    def get_service(self, service_type: Type[T]) -> T:
        """Get a service instance.
        
        Args:
            service_type: Type of service to retrieve
            
        Returns:
            Service instance
        """
        # Check if singleton exists
        if service_type in self._singletons:
            return self._singletons[service_type]
        
        # Check if factory exists
        if service_type in self._factories:
            factory = self._factories[service_type]
            
            # Inspect factory to get dependencies
            sig = inspect.signature(factory)
            dependencies = {}
            
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                
                # Try to resolve dependency from container
                if param.annotation != inspect.Parameter.empty:
                    try:
                        dependencies[param_name] = self.get_service(param.annotation)
                    except Exception as e:
                        logger.warning(f"Could not resolve dependency {param_name}: {e}")
                        if param.default != inspect.Parameter.empty:
                            dependencies[param_name] = param.default
                        else:
                            raise ValueError(f"Missing required dependency: {param_name}")
            
            # Create instance with resolved dependencies
            instance = factory(**dependencies)
            
            # Cache if it's a singleton factory
            if service_type not in self._factories or self._is_singleton_factory(factory):
                self._singletons[service_type] = instance
            
            return instance
        
        raise ValueError(f"Service not registered: {service_type.__name__}")
    
    def _is_singleton_factory(self, factory: Callable) -> bool:
        """Check if factory should be treated as singleton.
        
        Args:
            factory: Factory function
            
        Returns:
            True if singleton, False otherwise
        """
        # Simple heuristic: if factory name contains "singleton" or "create"
        factory_name = factory.__name__.lower()
        return "singleton" in factory_name or "create" in factory_name
    
    def clear(self) -> None:
        """Clear all registered services."""
        self._services.clear()
        self._factories.clear()
        self._singletons.clear()
        logger.info("Service container cleared")


class ServiceLocator(IServiceProvider):
    """Service locator for accessing services globally (use sparingly)."""
    
    _instance: Optional['ServiceLocator'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._container: Optional[ServiceContainer] = None
        return cls._instance
    
    def set_container(self, container: ServiceContainer) -> None:
        """Set the service container.
        
        Args:
            container: Service container instance
        """
        self._container = container
        logger.info("ServiceLocator container set")
    
    def get_service(self, service_type: Type[T]) -> T:
        """Get a service instance.
        
        Args:
            service_type: Type of service to retrieve
            
        Returns:
            Service instance
        """
        if self._container is None:
            raise RuntimeError("Service container not set")
        
        return self._container.get_service(service_type)
    
    @classmethod
    def reset(cls) -> None:
        """Reset the service locator."""
        cls._instance = None
        logger.info("ServiceLocator reset")


# Global service locator instance
service_locator = ServiceLocator()

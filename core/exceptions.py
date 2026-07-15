"""Domain exceptions for predictable structured error handling."""


class PlatformError(Exception):
    """Base error for platform failures."""


class ConfigurationError(PlatformError):
    """Raised when required configuration is invalid or missing."""


class PluginLoadError(PlatformError):
    """Raised when scenario or detector plugin loading fails."""


class TargetExecutionError(PlatformError):
    """Raised when a target request fails after retries."""


class DetectorError(PlatformError):
    """Raised when detector execution fails."""


class ScanCancelledError(PlatformError):
    """Raised when a running scan is cancelled by user request."""


"""Typed failures exposed by project boundaries."""


class ShopeeMatchError(RuntimeError):
    """Base error for actionable project failures."""


class ConfigurationError(ShopeeMatchError):
    """Raised when a versioned configuration is missing or invalid."""


class ContractError(ShopeeMatchError):
    """Raised when an online or batch contract is violated."""


class FixtureError(ShopeeMatchError):
    """Raised when the Phase 0 smoke fixture is incomplete or inconsistent."""


class DataValidationError(ShopeeMatchError):
    """Raised when critical Phase 1 data-quality findings block publication."""


class OutputConflictError(ShopeeMatchError):
    """Raised when a versioned output path already contains different content."""

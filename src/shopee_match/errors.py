"""Typed failures exposed by project boundaries."""


class ShopeeMatchError(RuntimeError):
    """Base error for actionable project failures."""


class ConfigurationError(ShopeeMatchError):
    """Raised when a versioned configuration is missing or invalid."""


class ContractError(ShopeeMatchError):
    """Raised when an online or batch contract is violated."""


class FixtureError(ShopeeMatchError):
    """Raised when the Phase 0 smoke fixture is incomplete or inconsistent."""

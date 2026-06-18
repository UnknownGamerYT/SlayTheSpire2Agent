"""Shared simulator exceptions."""


class STS2SimError(Exception):
    """Base error for simulator failures."""


class IllegalActionError(STS2SimError):
    """Raised when an action is invalid for the current state."""


class SourceDataError(STS2SimError):
    """Raised when source data is missing, malformed, or stale."""


class ContentNotImplementedError(STS2SimError):
    """Raised when source content exists but has no executable handler."""


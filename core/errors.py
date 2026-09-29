"""Application error vocabulary independent of HTTP transport."""

from __future__ import annotations


class BraindError(Exception):
    """Base class for failures that can be safely shown to API clients."""

    code = "braind_error"
    public_message = "The request could not be completed."
    status_code = 500


class ValidationError(BraindError):
    """Raised when a request is semantically invalid."""

    code = "validation_error"
    public_message = "The request could not be validated."
    status_code = 400


class NotFoundError(BraindError):
    """Raised when a requested Braind resource does not exist."""

    code = "not_found"
    public_message = "The requested resource was not found."
    status_code = 404


class ServiceUnavailableError(BraindError):
    """Raised when a required local service has not been initialized."""

    code = "service_unavailable"
    public_message = "A required service is currently unavailable."
    status_code = 503


class UpstreamServiceError(BraindError):
    """Raised when a required upstream service cannot complete an operation."""

    code = "upstream_error"
    public_message = "An upstream service could not complete the request."
    status_code = 502


class DatabaseInitializationError(BraindError):
    """Raised when required database initialization cannot complete."""

    code = "database_initialization_error"
    public_message = "The database could not be initialized."
    status_code = 500


class PolicyDeniedError(BraindError):
    """Raised when the governance policy engine blocks an action (Giai đoạn 5).

    The message carries the rule's reason so the client sees *why* — the
    decision is never silent.
    """

    code = "policy_denied"
    public_message = "The action was blocked by governance policy."
    status_code = 403

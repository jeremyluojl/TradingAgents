"""Exceptions that tell ``route_to_vendor`` to try the next data vendor."""


class VendorError(Exception):
    """A vendor could not serve the request; the router should fall back."""


class VendorNoDataError(VendorError):
    """The vendor responded but had no usable data for the request."""


class VendorUnavailableError(VendorError):
    """The vendor is unreachable, unsupported for this input, or misconfigured."""

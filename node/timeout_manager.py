# backwards compat shim — implementation lives in node.resource_manager
from node.resource_manager import JobTimeoutError, TimeoutManager  # noqa: F401

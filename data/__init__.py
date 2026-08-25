from .registry import build, register
from . import ucr_loader   # ensure UCR/UEA dataset registrations run on import
from . import tsb_loader   # ensure TSB-StreamingAD dataset registrations run on import

__all__ = ["build", "register"]

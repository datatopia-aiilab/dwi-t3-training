"""
Model Factory - Backward Compatibility Wrapper
This file maintains backward compatibility while redirecting to the new models/ directory

⚠️  DEPRECATED: This file is kept for backward compatibility only.
    New code should import from models/ directory directly:
    
    from models import get_model
    model = get_model(config)
"""

import warnings

# Import from new models directory
from models import get_model, get_attention_unet

# Show deprecation warning
warnings.warn(
    "Importing from model.py is deprecated. "
    "Please update your imports to:\n"
    "  from models import get_model\n"
    "  model = get_model(config)",
    DeprecationWarning,
    stacklevel=2
)

# Re-export for compatibility
__all__ = ['get_model', 'get_attention_unet']

try:
    import mmcv
    _mmcv_available = True
except ImportError:
    _mmcv_available = False

from .version import __version__, version_info

__all__ = ['__version__', 'version_info']

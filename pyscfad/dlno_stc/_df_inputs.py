"""Select direct fragment integrals or resident H reuse."""
import os


def cache_mode():
    mode = os.environ.get('PYSCFAD_LNO_SHARED_LOCAL_DF_CACHE_MODE', 'direct').lower()
    if mode not in ('direct', 'half'):
        raise ValueError('PYSCFAD_LNO_SHARED_LOCAL_DF_CACHE_MODE must be direct or half')
    return mode


def configure_cache(scope, static=None):
    """Configure optional H reuse for domain MP2 and LIS consumers of the same frame."""
    if static is not None:
        from .lis import _validate_lis_space
        _validate_lis_space(static.lis_static.lis_space)
    mode = cache_mode()
    scope.receipt['cache_mode'] = mode
    if mode == 'half':
        from pyscfad.lno._df_half import HalfTransformedDFCache
        scope.half = HalfTransformedDFCache(scope)

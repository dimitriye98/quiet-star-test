from .config import DEFAULT_CONFIG, resolve_torch_dtype, save_config, load_config


def __getattr__( name ):
	from .thought_model import __all__ as _all
	if name in _all:
		from . import thought_model
		globals().update( { n: getattr( thought_model, n ) for n in _all } )
		return globals()[ name ]
	raise AttributeError( f"module {__name__!r} has no attribute {name!r}" )
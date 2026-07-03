import importlib

# symbol -> submodule that defines it
_EXPORTS = {
    "clean_text_for_embedding": "sentence_filter",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = importlib.import_module(f".{module}", __name__)
    return getattr(mod, name)


def __dir__():
    return sorted(_EXPORTS)

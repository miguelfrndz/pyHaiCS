import os, sys
from contextlib import ContextDecorator

class _HiddenPrintsMeta(type):
    def __call__(cls, *args, **kwargs):
        # If used as @HiddenPrints without parentheses
        if len(args) == 1 and callable(args[0]) and not kwargs:
            func = args[0]
            instance = cls()
            return instance(func)
        return super().__call__(*args, **kwargs)

class HiddenPrints(ContextDecorator, metaclass = _HiddenPrintsMeta):
    """
    Context manager and decorator to suppress print statements during tests.
    Can be used as:
        - @HiddenPrints
        - @HiddenPrints()
        - with HiddenPrints():
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._devnull = open(os.devnull, 'w')
        sys.stdout = self._devnull
        sys.stderr = self._devnull

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        self._devnull.close()
        return False
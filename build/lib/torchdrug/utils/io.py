import os
import sys
import ast
import tempfile
from contextlib import contextmanager

from rdkit import RDLogger


def input_choice(prompt, choice=("y", "n")):
    """
    Print a prompt on the command line and wait for a choice.

    Parameters:
         prompt (str): prompt string
         choice (tuple of str, optional): candidate choices
    """
    prompt = "%s (%s)" % (prompt, "/".join(choice))
    choice = set([c.lower() for c in choice])
    result = input(prompt)
    while result.lower() not in choice:
        result = input(prompt)
    return result


def literal_eval(string):
    """
    Evaluate an expression into a Python literal structure.
    """
    try:
        return ast.literal_eval(string)
    except (ValueError, SyntaxError):
        return string


@contextmanager
def no_rdkit_log():
    """
    Context manager to suppress all rdkit loggings.
    """
    RDLogger.DisableLog("rdApp.*")
    yield
    RDLogger.EnableLog("rdApp.*")


class CaptureStdIO(object):

    def __init__(self, stdout=True, stderr=False):
        self.stdout = stdout
        self.stderr = stderr
        self.file = tempfile.TemporaryFile("w+")

    def __enter__(self):
        if self.stdout:
            stdout_fd = sys.stdout.fileno()
            self.stdout_fd = os.dup(stdout_fd)
            os.dup2(self.file.fileno(), stdout_fd)
        if self.stderr:
            stderr_fd = sys.stderr.fileno()
            self.stderr_fd = os.dup(stderr_fd)
            os.dup2(self.file.fileno(), stderr_fd)
        return self

    def __exit__(self, type, value, traceback):
        if self.stdout:
            os.dup2(self.stdout_fd, sys.stdout.fileno())
            os.close(self.stdout_fd)
        if self.stderr:
            os.dup2(self.stderr_fd, sys.stderr.fileno())
            os.close(self.stderr_fd)
        self.file.seek(0)
        self.content = self.file.read().rstrip()
        self.file.close()


def capture_rdkit_log():
    """
    Context manager to capture all rdkit loggings.

    Example::

        >>> with utils.capture_rdkit_log() as log:
        >>>     ...
        >>> print(log.content)
    """
    return CaptureStdIO(True, True)
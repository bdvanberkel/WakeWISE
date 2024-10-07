import os

from WakeWISE.utils.globals import LOG_LEVEL, LOGGER_COLORS, LOGGER_CONFIG


def log(msg : str, origin : str = None, log_level : LOG_LEVEL = LOG_LEVEL.INFO) -> None:

    if log_level < os.environ['WAKEWISE_LOG_LEVEL']:
        return

    origin = "[" + origin + "]" if origin else ""

    print(LOGGER_COLORS.OKGREEN + "[" + LOGGER_CONFIG.SCRIPT_NAME + "]" + origin + ": " + msg + LOGGER_COLORS.ENDC)

def warn(msg : str, origin : str = None, log_level : LOG_LEVEL = LOG_LEVEL.WARNING) -> None:

    if log_level < os.environ['WAKEWISE_LOG_LEVEL']:
        return

    origin = "[" + origin + "]" if origin else ""

    print(LOGGER_COLORS.WARNING + "[" + LOGGER_CONFIG.SCRIPT_NAME + "]" + origin + ": " + msg + LOGGER_COLORS.ENDC)

def error(msg : str, origin : str = None, log_level : LOG_LEVEL = LOG_LEVEL.ERROR) -> None:

    if log_level < os.environ['WAKEWISE_LOG_LEVEL']:
        return

    origin = "[" + origin + "]" if origin else ""

    print(LOGGER_COLORS.FAIL + "[" + LOGGER_CONFIG.SCRIPT_NAME + "]" + origin + ": " + msg + LOGGER_COLORS.ENDC)

def note(msg : str, origin : str = None, log_level : LOG_LEVEL = LOG_LEVEL.INFO) -> None:

    if log_level < os.environ['WAKEWISE_LOG_LEVEL']:
        return

    origin = "[" + origin + "]" if origin else ""

    print(LOGGER_COLORS.OKCYAN + "[" + LOGGER_CONFIG.SCRIPT_NAME + "]" + origin + ": " + msg + LOGGER_COLORS.ENDC)
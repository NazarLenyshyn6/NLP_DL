import os
import logging

LOG_DIR = '/Users/nazarlenisin/Desktop/DL_LAB2/Logs'
LOG_FILE = os.path.join(LOG_DIR, 'project_logs.log')
LOGGING_FORMAT = '[%(asctime)s | %(name)s | %(levelname)s] -> %(message)s'

os.makedirs(LOG_DIR, exist_ok=True)


logging.basicConfig(level=logging.INFO,
                    format=LOGGING_FORMAT,
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler(LOG_FILE)])

logger = logging.getLogger('dl_lab2_logger')

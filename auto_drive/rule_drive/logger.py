import logging
import os

if not os.path.exists('/log'):
    os.makedirs('/log')

logging.basicConfig(level=logging.INFO, filename='/log/bot.log', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('socketio').setLevel(logging.ERROR)
logging.getLogger('engineio').setLevel(logging.ERROR)

def get_logger(name):
    logger = logging.getLogger(name)
    return logger
# Not required to run the app, but very useful for debugging, monitoring, and production logs.

# Without this, you might not see debug messages (since the default level is WARNING).

import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
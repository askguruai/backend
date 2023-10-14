# https://pawamoy.github.io/posts/unify-logging-for-a-gunicorn-uvicorn-app/

import logging
import sys

from gunicorn.app.base import BaseApplication
from gunicorn.glogging import Logger
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from utils import CONFIG

LOG_LEVEL = logging.getLevelName(CONFIG["app"]["log_level"].upper())


class RequestLoggerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # logger.info(f"{request.client.host}:{request.client.port} - {request.method} {request.url.path} {request.url.query} - {request.headers.get('user-agent', 'N/A')}")
        logger.info(
            f"{request.client.host}:{request.client.port} - {request.method} {request.url.path} {request.url.query}"
        )
        response = await call_next(request)
        return response


class InterceptHandler(logging.Handler):
    def emit(self, record):
        # get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # find caller from where originated the logged message
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


class StubbedGunicornLogger(Logger):
    def setup(self, cfg):
        handler = logging.NullHandler()
        self.error_logger = logging.getLogger("gunicorn.error")
        self.error_logger.addHandler(handler)
        self.access_logger = logging.getLogger("gunicorn.access")
        self.access_logger.addHandler(handler)
        self.error_logger.setLevel(LOG_LEVEL)
        self.access_logger.setLevel(LOG_LEVEL)


class StandaloneApplication(BaseApplication):
    """Our Gunicorn application."""

    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super().__init__()

    def load_config(self):
        config = {key: value for key, value in self.options.items() if key in self.cfg.settings and value is not None}
        for key, value in config.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application


def run_gunicorn_loguru(app, options):
    # `proxy_headers` and `forwarded_allow_ips` are set to "*" to allow
    # the application to receive real IP addresses from nginx.
    #
    # In FastAPI it works seamlessly via request.client.host.
    #
    # In nginx we should set following headers:
    # proxy_set_header Host $host;
    # proxy_set_header X-Real-IP $remote_addr;
    # proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    options |= {
        "accesslog": "-",
        "errorlog": "-",
        "worker_class": "uvicorn.workers.UvicornWorker",
        # "worker_connections": 1000,
        "logger_class": StubbedGunicornLogger,
        "proxy_headers": True,
        "forwarded_allow_ips": "*",
    }
    intercept_handler = InterceptHandler()
    # logging.basicConfig(handlers=[intercept_handler], level=LOG_LEVEL)
    # logging.root.handlers = [intercept_handler]
    logging.root.setLevel(LOG_LEVEL)

    seen = set()
    for name in [
        *logging.root.manager.loggerDict.keys(),
        "gunicorn",
        "gunicorn.access",
        "gunicorn.error",
        "uvicorn",
        "uvicorn.access",
        "uvicorn.error",
    ]:
        if name not in seen:
            seen.add(name.split(".")[0])
            logging.getLogger(name).handlers = [intercept_handler]

    logger.configure(handlers=[{"sink": sys.stdout, "serialize": False}])

    StandaloneApplication(app, options).run()

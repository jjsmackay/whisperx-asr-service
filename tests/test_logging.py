import logging
import logging.config
import importlib


def test_app_logger_is_info_under_uvicorn_dict_config():
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "loggers": {"app": {"level": "WARNING"}},
    })
    import app.main
    importlib.reload(app.main)
    assert logging.getLogger("app").getEffectiveLevel() == logging.INFO

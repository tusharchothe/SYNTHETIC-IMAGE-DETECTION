import importlib


def test_package_modules_import():
    modules = [
        "aidetect",
        "aidetect.config",
        "aidetect.data",
        "aidetect.models",
        "aidetect.training",
        "aidetect.evaluation",
        "aidetect.inference",
        "aidetect.cli",
    ]
    for module_name in modules:
        importlib.import_module(module_name)

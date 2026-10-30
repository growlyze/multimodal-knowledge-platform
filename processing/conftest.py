# telemetry/smoke_test.py matches pytest's *_test.py pattern but is a script run inside the
# phoenix container (see its docstring), not a test module.
collect_ignore = ["telemetry/smoke_test.py"]

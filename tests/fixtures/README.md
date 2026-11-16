# Regression fixtures only

The JSON files under `config/` contain normalized geometry and policy values needed to reproduce route, barrier, acknowledgement and continuation regressions. Hardware examples are synthetic; comments identifying incidents or installations have been removed. There are no camera images, credentials, geographical coordinates, registration poses or learned barrier/empty-road templates.

These fixtures are **not live configuration** and are not a recommended layout for another camera. `tools/run_tests.py` materializes them only in a temporary test checkout. Configure your own installation from the generic runtime example.

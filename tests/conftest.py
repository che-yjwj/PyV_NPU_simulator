import sys
import os

# Add the 'py-v' directory to the Python path to resolve the ModuleNotFoundError
# This allows the tests to find the 'pyv' module, which is a local dependency.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'py-v')))

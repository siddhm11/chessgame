"""Test package marker.

Makes `tests` a regular package so `from tests.generate_fixtures import
generate` in conftest.py always resolves to this directory, even when a
dependency (e.g. an ML toolkit) installs an unrelated top-level `tests`
package into site-packages that would otherwise shadow it.
"""

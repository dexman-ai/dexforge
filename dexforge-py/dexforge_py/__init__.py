"""dexforge_py Python package structure.

This allows dexforge_py to be imported as a package with submodules.
The actual compiled extension is loaded when this package is imported.
"""

# The maturin-built extension will be available as dexforge_py.*
# We don't need to do anything here since maturin handles the module setup

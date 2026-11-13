"""Example ASSIST plugins as an installable package.

The three modules are the same files as the loose ones in `examples/plugins/`. Installed, each is reachable by the
entry-point name this package declares in the group `assist.plugins` (see pyproject.toml):

    plugins:
      - pretty_gfsk      # the signal kind `pretty_gfsk`  (assist_examples.pretty_gfsk)
      - tethered         # the platform `tethered`        (assist_examples.jitter_platform)
      - tcp              # the sink `tcp`                 (assist_examples.tcp_sink)

Nothing is imported here: a scenario that names one plugin loads only that module.
"""

__version__ = "0.1.0"

"""Ship tool handlers.

Three thin handlers that delegate to an injected ``SimulatorClient``:
``set_heading``, ``set_engine_telegraph``, ``get_ship_state``. Handlers carry
only input validation; all state lives in the ``SimulatorClient``.
"""

from __future__ import annotations


def register_ship_tools(llm, simulator) -> None:
    """Register the three ship tool handlers on the LLM service.

    The single ``simulator`` (a ``SimulatorClient``) is closed over by all
    three handlers.
    """
    raise NotImplementedError("tools.ship.register_ship_tools is a scaffold stub")

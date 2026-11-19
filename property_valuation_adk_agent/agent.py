"""
ADK Web / CLI Entrypoint (`adk web` / `adk run property_valuation_adk_agent`)
Exports `root_agent` pointing to `orchestrator_agent`.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adk_valuation_agents import orchestrator_agent as root_agent

__all__ = ["root_agent"]

"""Zimmerman Simulation Toolkit.

A generalized library for interrogating black-box simulators. Given any
simulator that satisfies a simple protocol -- run(params) -> result_dict,
param_spec() -> bounds -- the toolkit answers the questions that matter:
what inputs drive outcomes, where are the tipping points, does the system
do what you think it does, and where does it break.

Based on Julia Zimmerman's 2025 PhD dissertation at the University of
Vermont: "Locality, Relation, and Meaning Construction in Language, as
Implemented in Humans and Large Language Models (LLMs)."

Generalized from domain-specific code in the how-to-live-much-longer
project (Cramer 2025 mitochondrial aging simulator).

Modules:
    base        -- Simulator protocol and wrapper types
    sobol       -- Saltelli sampling + Sobol sensitivity indices
    pds         -- Power-Danger-Structure dimension mapper (Zimmerman §4.6.4; Dodds et al. 2023)
    posiwid     -- POSIWID alignment auditor (intended vs actual; Beer 1974; Zimmerman §3.5.2)
    prompts     -- Diegetic prompt builder for LLM-mediated design (Zimmerman §2.2.3, §3.5.3, §4.7.6)
    contrastive -- Contrastive scenario generation (minimal outcome flips)
    falsifier   -- Systematic falsification and assumption testing
"""

from zimmerman.base import Simulator, SimulatorWrapper
from zimmerman.sobol import sobol_sensitivity, saltelli_sample, rescale_samples, sobol_indices
from zimmerman.pds import PDSMapper
from zimmerman.posiwid import POSIWIDAuditor
from zimmerman.prompts import PromptBuilder
from zimmerman.contrastive import ContrastiveGenerator
from zimmerman.falsifier import Falsifier

__version__ = "0.1.0"

__all__ = [
    "Simulator",
    "SimulatorWrapper",
    "sobol_sensitivity",
    "saltelli_sample",
    "rescale_samples",
    "sobol_indices",
    "PDSMapper",
    "POSIWIDAuditor",
    "PromptBuilder",
    "ContrastiveGenerator",
    "Falsifier",
]

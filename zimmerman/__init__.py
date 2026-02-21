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
    contrast_set_generator -- Structured edit-space contrast sets (TALOT/OTTITT harness)
    falsifier   -- Systematic falsification and assumption testing
    relation_graph_extractor -- Meaning-from-relations multigraph extraction (Zimmerman §2-3)
    locality_profiler    -- Locality profiling via manipulation sweeps (Zimmerman §3.5, §4.6)
    prompt_receptive_field -- Feature attribution over input segments via Sobol (Zimmerman §4.6, §4.7)
    diegeticizer -- Reversible translation between parameter vectors and narrative descriptions
    supradiegetic_benchmark -- Standardized form-vs-meaning battery (diegeticization gain)
    token_extispicy -- Token fragmentation hazard surface analysis (Zimmerman §3.5.3)
    meaning_construction_dashboard -- Unified aggregator for multi-tool reports
    trajectory_metrics   -- State-space path metrics for ODE simulator trajectories
    output_schema        -- Shared JSON envelope for cross-simulator output interop
"""

from zimmerman.base import Simulator, SimulatorWrapper
from zimmerman.sobol import sobol_sensitivity, saltelli_sample, rescale_samples, sobol_indices
from zimmerman.pds import PDSMapper
from zimmerman.posiwid import POSIWIDAuditor
from zimmerman.prompts import PromptBuilder
from zimmerman.contrastive import ContrastiveGenerator
from zimmerman.contrast_set_generator import ContrastSetGenerator
from zimmerman.falsifier import Falsifier
from zimmerman.relation_graph_extractor import RelationGraphExtractor
from zimmerman.locality_profiler import LocalityProfiler
from zimmerman.prompt_receptive_field import PromptReceptiveField
from zimmerman.diegeticizer import Diegeticizer
from zimmerman.supradiegetic_benchmark import SuperdiegeticBenchmark
from zimmerman.token_extispicy import TokenExtispicyWorkbench
from zimmerman.meaning_construction_dashboard import MeaningConstructionDashboard
from zimmerman.trajectory_metrics import trajectory_metrics, TrajectoryMetricsProfiler
from zimmerman.output_schema import SimulatorOutput, validate_output, compare_outputs, NumpyEncoder
from zimmerman.visualizations import sobol_class_profile_matrix, plot_sobol_class_profiles

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
    "ContrastSetGenerator",
    "Falsifier",
    "RelationGraphExtractor",
    "LocalityProfiler",
    "PromptReceptiveField",
    "Diegeticizer",
    "SuperdiegeticBenchmark",
    "TokenExtispicyWorkbench",
    "MeaningConstructionDashboard",
    "trajectory_metrics",
    "TrajectoryMetricsProfiler",
    "SimulatorOutput",
    "validate_output",
    "compare_outputs",
    "NumpyEncoder",
    "sobol_class_profile_matrix",
    "plot_sobol_class_profiles",
]

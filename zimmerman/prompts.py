"""Diegetic prompt builder for LLM-mediated simulation design.

Generalized from how-to-live-much-longer/prompt_templates.py. Builds
prompts that ask an LLM to generate simulator parameter values.

Three prompt styles:
    NumericPrompt: Straightforward parameter request with bounds.
        Lists each parameter with its (min, max) range and asks for values.

    DiegeticPrompt: Narrative/contextual prompt that references the
        simulation's own state. Embeds parameters in a domain narrative
        rather than presenting them as raw numbers. Based on Zimmerman
        (2025) Ch. 2-3: LLMs handle semantic content better than
        structural/numeric content.

    ContrastivePrompt: "What would a cautious vs aggressive agent choose?"
        Exploits TALOT/OTTITT meaning-from-contrast (Zimmerman Ch. 5).
        Generates opposing parameter sets that bracket the problem.

All prompts include param_spec bounds in the output so the LLM knows
the valid ranges.

Reference:
    Zimmerman, J.W. (2025). "Locality, Relation, and Meaning Construction
    in Language, as Implemented in Humans and Large Language Models (LLMs)."
    PhD dissertation, University of Vermont.
"""

from __future__ import annotations


class PromptBuilder:
    """Builds prompts for LLM-mediated parameter generation.

    Constructs prompt strings that instruct an LLM to generate parameter
    values for a simulator, in various styles (numeric, diegetic,
    contrastive).

    Args:
        simulator: Any Simulator-compatible object with param_spec().
        context: Optional dict of additional context to include in
            prompts (e.g., domain knowledge, constraints, previous
            results). Keys are used as section headers.

    Example:
        builder = PromptBuilder(my_simulator, context={"domain": "robotics"})
        prompt = builder.build_numeric("Design a fast robot")
        # Send prompt to LLM...
    """

    def __init__(self, simulator, context: dict | None = None):
        self.simulator = simulator
        self._spec = simulator.param_spec()
        self.context = context or {}

    def _format_param_spec(self) -> str:
        """Format the parameter specification as a readable string."""
        lines = []
        for name, (lo, hi) in self._spec.items():
            lines.append(f"  {name}: ({lo}, {hi})")
        return "\n".join(lines)

    def _format_param_json_template(self) -> str:
        """Format param names as a JSON template string."""
        keys = list(self._spec.keys())
        inner = ", ".join(f'"{k}": _' for k in keys)
        return "{" + inner + "}"

    def _format_context(self) -> str:
        """Format optional context as a string block."""
        if not self.context:
            return ""
        lines = ["\n=== CONTEXT ==="]
        for key, val in self.context.items():
            lines.append(f"\n{key}:")
            lines.append(f"  {val}")
        return "\n".join(lines)

    def build_numeric(self, scenario: str) -> str:
        """Build a numeric-style prompt: straightforward parameter request.

        Lists each parameter with its valid range and asks the LLM to
        produce a JSON object with values for each parameter.

        Args:
            scenario: Description of the scenario or design goal.

        Returns:
            Prompt string ready to send to an LLM.
        """
        spec_str = self._format_param_spec()
        template = self._format_param_json_template()
        context_str = self._format_context()

        prompt = f"""\
You are a simulation design specialist. Given the scenario below,
choose parameter values that best achieve the described goal.

SCENARIO:
{scenario}

PARAMETERS (name: (min, max)):
{spec_str}

Choose a value for each parameter within its valid range.
{context_str}

Output a JSON object with ALL parameter keys:
{template}

Brief reasoning (1-2 sentences), then ONLY the JSON object."""
        return prompt

    def build_diegetic(
        self,
        scenario: str,
        state_description: str | None = None,
    ) -> str:
        """Build a diegetic-style prompt: narrative/contextual.

        Embeds the parameter choices in a narrative context rather than
        presenting them as raw numbers. This aligns with Zimmerman (2025)
        Ch. 2-3: LLMs construct meaning from distributional semantics
        (diegetic content), not from structural form.

        Args:
            scenario: Description of the scenario or design goal.
            state_description: Optional description of the current
                simulation state (e.g., "The system is currently running
                at 60% efficiency with high damage levels").

        Returns:
            Prompt string ready to send to an LLM.
        """
        spec_str = self._format_param_spec()
        template = self._format_param_json_template()
        context_str = self._format_context()

        state_block = ""
        if state_description:
            state_block = f"""
=== CURRENT STATE ===
{state_description}
"""

        # Build descriptive parameter section
        param_descriptions = []
        for name, (lo, hi) in self._spec.items():
            midpoint = (lo + hi) / 2.0
            param_descriptions.append(
                f"  - {name}: How strongly should this be set? "
                f"(ranges from {lo} [minimum] to {hi} [maximum], "
                f"midpoint = {midpoint:.2f})"
            )
        desc_str = "\n".join(param_descriptions)

        prompt = f"""\
You are an expert designing a simulation configuration. Consider the
following scenario as a real situation that you must respond to:

SCENARIO:
{scenario}
{state_block}
Think about each design decision as a judgment call:

{desc_str}
{context_str}

Consider the tradeoffs carefully. What are the most important parameters
for this scenario? Which parameters interact with each other?

After your reasoning, output a JSON object with ALL parameter keys.
Valid ranges:
{spec_str}

Format: {template}"""
        return prompt

    def build_contrastive(
        self,
        scenario: str,
        agent_a: str = "cautious",
        agent_b: str = "aggressive",
    ) -> str:
        """Build a contrastive-style prompt: two opposing agents.

        Asks the LLM to generate TWO parameter sets: one from a cautious
        perspective and one from an aggressive perspective. This exploits
        TALOT/OTTITT meaning-from-contrast (Zimmerman Ch. 5) and forces
        the LLM to think more carefully about parameter choices.

        Args:
            scenario: Description of the scenario or design goal.
            agent_a: Description or name of the first (conservative) agent.
            agent_b: Description or name of the second (aggressive) agent.

        Returns:
            Prompt string ready to send to an LLM.
        """
        spec_str = self._format_param_spec()
        template = self._format_param_json_template()
        context_str = self._format_context()

        prompt = f"""\
You are mediating between two experts who disagree about the best
approach to this scenario. You must present BOTH positions.

SCENARIO:
{scenario}

=== EXPERT A: {agent_a.upper()} ===
This expert prioritizes safety, stability, and minimal intervention.
They prefer conservative parameter choices and avoid extremes.

=== EXPERT B: {agent_b.upper()} ===
This expert believes the situation requires maximum response. They
push parameters hard and accept higher risk for higher reward.

PARAMETERS (name: (min, max)):
{spec_str}
{context_str}

For EACH expert, provide a parameter configuration as a JSON object.
In 2-3 sentences, explain what each expert would argue and WHY they
disagree.

Output TWO JSON objects labeled "{agent_a}" and "{agent_b}":
{{"{agent_a}": {template}, "{agent_b}": {template}}}"""
        return prompt

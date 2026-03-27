"""
JARVIS Iteration 17: Superintelligence Core
Meta-cognitive engine with uncertainty quantification, cognitive bias correction,
adversarial thinking, and calibrated confidence — superhuman judgment system.
"""
import asyncio
import json
import math
import re
import statistics
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class CognitiveBias(Enum):
    CONFIRMATION_BIAS = "confirmation_bias"
    ANCHORING_BIAS = "anchoring_bias"
    AVAILABILITY_HEURISTIC = "availability_heuristic"
    DUNNING_KRUGER = "dunning_kruger"
    SUNK_COST_FALLACY = "sunk_cost_fallacy"
    RECENCY_BIAS = "recency_bias"
    SURVIVORSHIP_BIAS = "survivorship_bias"
    OVERCONFIDENCE = "overconfidence"
    GROUPTHINK = "groupthink"
    FRAMING_EFFECT = "framing_effect"


class ReasoningMode(Enum):
    FAST = "fast"           # System 1: intuitive
    SLOW = "slow"           # System 2: deliberate
    ADVERSARIAL = "adversarial"  # Devil's advocate
    PROBABILISTIC = "probabilistic"  # Bayesian
    COUNTERFACTUAL = "counterfactual"  # What-if
    DIALECTICAL = "dialectical"  # Thesis-antithesis-synthesis


@dataclass
class ThoughtChain:
    """A complete chain of reasoning."""
    question: str
    mode: ReasoningMode
    premises: list[str] = field(default_factory=list)
    reasoning_steps: list[str] = field(default_factory=list)
    conclusion: str = ""
    confidence: float = 0.0
    uncertainty: float = 0.0
    biases_detected: list[CognitiveBias] = field(default_factory=list)
    biases_corrected: bool = False
    adversarial_challenges: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    quality_score: float = 0.0


@dataclass
class BeliefState:
    """Calibrated belief about a proposition."""
    proposition: str
    probability: float  # 0.0 - 1.0
    confidence_interval: tuple[float, float] = (0.0, 1.0)
    evidence_for: list[str] = field(default_factory=list)
    evidence_against: list[str] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)
    update_count: int = 0


class SuperintelligenceCore:
    """
    Meta-cognitive superintelligence engine.

    Key capabilities:
    1. Dual-process thinking (fast intuition + slow reasoning)
    2. Cognitive bias detection and auto-correction
    3. Uncertainty quantification and calibration
    4. Adversarial self-questioning
    5. Bayesian belief updating
    6. Counterfactual reasoning
    7. Strategic long-term planning
    8. Cross-domain insight transfer
    9. Metacognitive monitoring
    10. Truth-seeking over comfort
    """

    COGNITIVE_BIASES_PROMPTS = {
        CognitiveBias.CONFIRMATION_BIAS: "Am I ignoring evidence that contradicts my conclusion?",
        CognitiveBias.ANCHORING_BIAS: "Am I too influenced by the first information I received?",
        CognitiveBias.AVAILABILITY_HEURISTIC: "Am I overweighting easily recalled examples?",
        CognitiveBias.OVERCONFIDENCE: "Am I claiming more certainty than my evidence warrants?",
        CognitiveBias.RECENCY_BIAS: "Am I overemphasizing recent information vs historical patterns?",
        CognitiveBias.SURVIVORSHIP_BIAS: "Am I only looking at successes and ignoring failures?",
        CognitiveBias.SUNK_COST_FALLACY: "Am I continuing a path due to past investment vs future value?",
        CognitiveBias.GROUPTHINK: "Am I conforming to consensus at the expense of independent analysis?",
        CognitiveBias.FRAMING_EFFECT: "Is my answer influenced by how the question is framed?",
        CognitiveBias.DUNNING_KRUGER: "Am I overestimating my competence in an unfamiliar domain?",
    }

    def __init__(self, llm_manager):
        self.llm = llm_manager
        self.belief_system: dict[str, BeliefState] = {}
        self.thought_history: list[ThoughtChain] = []
        self.metacognitive_log: list[dict] = []
        self.performance_calibration: list[dict] = []  # Track prediction accuracy
        self.cognitive_profile = {
            "total_thoughts": 0,
            "biases_detected": 0,
            "biases_corrected": 0,
            "avg_confidence": 0.0,
            "calibration_score": 0.0,
            "adversarial_challenges": 0
        }
        self._running = False
        print("[SuperintelligenceCore] Metacognitive engine online")

    async def think(self, question: str, mode: ReasoningMode = ReasoningMode.SLOW) -> ThoughtChain:
        """
        Core thinking method with full metacognitive monitoring.
        """
        chain = ThoughtChain(question=question, mode=mode)
        start = time.time()

        if mode == ReasoningMode.FAST:
            chain = await self._fast_think(chain)
        elif mode == ReasoningMode.SLOW:
            chain = await self._slow_think(chain)
        elif mode == ReasoningMode.ADVERSARIAL:
            chain = await self._adversarial_think(chain)
        elif mode == ReasoningMode.PROBABILISTIC:
            chain = await self._probabilistic_think(chain)
        elif mode == ReasoningMode.COUNTERFACTUAL:
            chain = await self._counterfactual_think(chain)
        elif mode == ReasoningMode.DIALECTICAL:
            chain = await self._dialectical_think(chain)

        # Detect biases in the reasoning
        chain.biases_detected = await self._detect_biases(chain)

        # Auto-correct if biases found
        if chain.biases_detected:
            chain = await self._correct_biases(chain)
            chain.biases_corrected = True
            self.cognitive_profile["biases_corrected"] += 1

        # Calibrate confidence
        chain.confidence, chain.uncertainty = await self._calibrate_confidence(chain)

        # Quality score
        chain.quality_score = self._score_reasoning_quality(chain)

        # Update stats
        self.cognitive_profile["total_thoughts"] += 1
        self.cognitive_profile["biases_detected"] += len(chain.biases_detected)
        self.thought_history.append(chain)

        return chain

    async def _fast_think(self, chain: ThoughtChain) -> ThoughtChain:
        """System 1: Fast, intuitive thinking."""
        prompt = f"""Answer quickly and intuitively: {chain.question}

Give a direct, confident answer based on pattern recognition. 1-2 sentences."""

        response = await self.llm.generate_async(prompt, max_tokens=300, temperature=0.5)
        chain.conclusion = response or ""
        chain.reasoning_steps = ["Fast intuitive response"]
        chain.confidence = 0.65  # Base fast-think confidence
        return chain

    async def _slow_think(self, chain: ThoughtChain) -> ThoughtChain:
        """System 2: Deliberate, step-by-step reasoning."""
        prompt = f"""Think carefully and systematically about: {chain.question}

Step 1: Identify the core question and what's being asked
Step 2: List key facts and relevant knowledge
Step 3: Consider multiple perspectives
Step 4: Reason through the implications
Step 5: Check for logical fallacies
Step 6: Reach a well-supported conclusion

Provide detailed reasoning:"""

        response = await self.llm.generate_async(prompt, max_tokens=1500, temperature=0.2)
        if response:
            # Parse steps from response
            lines = response.split('\n')
            chain.reasoning_steps = [l for l in lines if l.strip()]
            chain.conclusion = lines[-1] if lines else response
        chain.confidence = 0.8
        return chain

    async def _adversarial_think(self, chain: ThoughtChain) -> ThoughtChain:
        """Devil's advocate: actively challenge the obvious answer."""
        # First get the obvious answer
        obvious_prompt = f"What is the most common/obvious answer to: {chain.question}"
        obvious = await self.llm.generate_async(obvious_prompt, max_tokens=300, temperature=0.3)

        # Then challenge it
        challenge_prompt = f"""Question: {chain.question}

The obvious answer is: {obvious}

As a devil's advocate, challenge this answer:
1. What assumptions does it make?
2. What evidence could refute it?
3. What alternative explanations exist?
4. What edge cases break it?
5. What would make the opposite true?

After challenging, what is the most defensible true answer?"""

        response = await self.llm.generate_async(challenge_prompt, max_tokens=1200, temperature=0.4)
        chain.adversarial_challenges = [obvious] if obvious else []
        chain.reasoning_steps = ["Obvious answer identified", "Adversarial challenge applied"]
        chain.conclusion = response or ""
        chain.confidence = 0.85  # Adversarial thinking produces more reliable answers
        return chain

    async def _probabilistic_think(self, chain: ThoughtChain) -> ThoughtChain:
        """Bayesian reasoning with explicit probability estimates."""
        prompt = f"""Apply Bayesian reasoning to: {chain.question}

1. Prior probability (before evidence): [state base rate]
2. Evidence for: [list key supporting evidence with likelihood ratios]
3. Evidence against: [list key counter-evidence with likelihood ratios]
4. Posterior probability after all evidence: [0.0-1.0]
5. Confidence interval: [lower, upper]
6. Key uncertainties: [what would change the estimate]

Final probabilistic assessment:"""

        response = await self.llm.generate_async(prompt, max_tokens=1000, temperature=0.2)

        # Try to extract probability from response
        prob_match = re.search(r'(?:posterior|probability)[:\s]+([0-9.]+)', response or "", re.IGNORECASE)
        if prob_match:
            try:
                chain.confidence = float(prob_match.group(1))
                if chain.confidence > 1.0:
                    chain.confidence /= 100.0
            except ValueError:
                chain.confidence = 0.7

        chain.reasoning_steps = ["Bayesian probability assessment"]
        chain.conclusion = response or ""
        return chain

    async def _counterfactual_think(self, chain: ThoughtChain) -> ThoughtChain:
        """What-if reasoning: explore alternate realities."""
        prompt = f"""Counterfactual analysis of: {chain.question}

Explore these what-if scenarios:
1. What if the opposite were true? What would change?
2. What if key assumptions were wrong?
3. What if this happened 10 years earlier / later?
4. What if resources were unlimited?
5. What if the worst case occurred?

Based on this analysis, what is most likely true and why?"""

        response = await self.llm.generate_async(prompt, max_tokens=1200, temperature=0.4)
        chain.reasoning_steps = ["Counterfactual scenario analysis"]
        chain.conclusion = response or ""
        chain.confidence = 0.75
        return chain

    async def _dialectical_think(self, chain: ThoughtChain) -> ThoughtChain:
        """Hegelian dialectic: thesis → antithesis → synthesis."""
        prompt = f"""Apply dialectical reasoning to: {chain.question}

THESIS (conventional position):
[State the mainstream/conventional answer]

ANTITHESIS (opposing position):
[State the strongest counter-argument]

SYNTHESIS (transcendent conclusion):
[Integrate both positions into a higher-order understanding that resolves the contradiction]

Final synthesis:"""

        response = await self.llm.generate_async(prompt, max_tokens=1200, temperature=0.3)

        # Parse thesis/antithesis/synthesis
        sections = {"THESIS": "", "ANTITHESIS": "", "SYNTHESIS": ""}
        current_section = None
        if response:
            for line in response.split('\n'):
                for section in sections:
                    if section in line.upper():
                        current_section = section
                        break
                if current_section and line.strip():
                    sections[current_section] += line + '\n'

        chain.premises = [sections.get("THESIS", ""), sections.get("ANTITHESIS", "")]
        chain.conclusion = sections.get("SYNTHESIS", response or "")
        chain.reasoning_steps = ["Thesis", "Antithesis", "Synthesis"]
        chain.confidence = 0.82
        return chain

    async def _detect_biases(self, chain: ThoughtChain) -> list[CognitiveBias]:
        """Detect cognitive biases in a reasoning chain."""
        if not chain.conclusion:
            return []

        detected = []
        text = chain.conclusion + ' '.join(chain.reasoning_steps)

        # Heuristic bias detection
        if re.search(r'\balways\b|\bnever\b|\beveryone\b|\bno one\b', text, re.IGNORECASE):
            detected.append(CognitiveBias.OVERCONFIDENCE)

        if re.search(r'\bobviously\b|\bclearly\b|\bof course\b|\bundoubtedly\b', text, re.IGNORECASE):
            detected.append(CognitiveBias.OVERCONFIDENCE)

        if re.search(r'\brecently\b|\blast year\b|\bnowadays\b|\bthese days\b', text, re.IGNORECASE):
            detected.append(CognitiveBias.RECENCY_BIAS)

        if re.search(r'\bsuccess(ful)?\b|\bwinner\b|\bachiev', text, re.IGNORECASE):
            if not re.search(r'\bfail\b|\bloss\b|\bunsuccessful\b', text, re.IGNORECASE):
                detected.append(CognitiveBias.SURVIVORSHIP_BIAS)

        # LLM-based deep bias detection for slow mode
        if chain.mode in (ReasoningMode.SLOW, ReasoningMode.ADVERSARIAL) and len(text) > 200:
            prompt = f"""Identify cognitive biases in this reasoning:

{text[:800]}

List only the bias names if present (comma-separated), or 'none':
Possible biases: {', '.join(b.value for b in CognitiveBias)}"""

            try:
                bias_response = await self.llm.generate_async(prompt, max_tokens=200, temperature=0.1)
                if bias_response and 'none' not in bias_response.lower():
                    for bias in CognitiveBias:
                        if bias.value.replace('_', ' ') in bias_response.lower() or bias.value in bias_response.lower():
                            if bias not in detected:
                                detected.append(bias)
            except Exception:
                pass

        self.cognitive_profile["biases_detected"] += len(detected)
        return detected

    async def _correct_biases(self, chain: ThoughtChain) -> ThoughtChain:
        """Re-reason with explicit bias corrections."""
        bias_corrections = [
            self.COGNITIVE_BIASES_PROMPTS[b] for b in chain.biases_detected
            if b in self.COGNITIVE_BIASES_PROMPTS
        ]

        correction_prompt = f"""Your previous reasoning about "{chain.question}" may contain biases.

Original conclusion: {chain.conclusion[:500]}

Bias correction checklist:
{chr(10).join(f'- {q}' for q in bias_corrections)}

Provide a bias-corrected, more balanced conclusion:"""

        try:
            corrected = await self.llm.generate_async(correction_prompt, max_tokens=800, temperature=0.2)
            if corrected:
                chain.conclusion = corrected
                chain.reasoning_steps.append("Bias correction applied")
        except Exception as e:
            pass

        return chain

    async def _calibrate_confidence(self, chain: ThoughtChain) -> tuple[float, float]:
        """
        Estimate calibrated confidence and uncertainty.
        Research shows LLMs are systematically overconfident.
        We apply a calibration correction.
        """
        base_confidence = chain.confidence

        # Calibration factors
        mode_factors = {
            ReasoningMode.FAST: 0.85,
            ReasoningMode.SLOW: 0.95,
            ReasoningMode.ADVERSARIAL: 0.92,
            ReasoningMode.PROBABILISTIC: 0.88,
            ReasoningMode.COUNTERFACTUAL: 0.80,
            ReasoningMode.DIALECTICAL: 0.90,
        }

        calibration_factor = mode_factors.get(chain.mode, 0.85)

        # Reduce confidence for each uncorrected bias
        if chain.biases_detected and not chain.biases_corrected:
            calibration_factor -= 0.05 * len(chain.biases_detected)

        # Increase confidence for adversarial challenges survived
        if chain.adversarial_challenges:
            calibration_factor += 0.03

        calibrated = max(0.1, min(0.99, base_confidence * calibration_factor))
        uncertainty = 1.0 - calibrated

        return calibrated, uncertainty

    def _score_reasoning_quality(self, chain: ThoughtChain) -> float:
        """Score the quality of a reasoning chain 0-1."""
        score = 0.5  # Base

        # More reasoning steps = higher quality
        score += min(0.2, len(chain.reasoning_steps) * 0.04)

        # Bias detection and correction
        if chain.biases_detected:
            if chain.biases_corrected:
                score += 0.15  # Found and corrected
            else:
                score -= 0.1   # Found but not corrected

        # Adversarial challenges considered
        if chain.adversarial_challenges:
            score += 0.1

        # Conclusion quality (length as proxy)
        if len(chain.conclusion) > 200:
            score += 0.05

        return max(0.0, min(1.0, score))

    async def update_belief(self, proposition: str, new_evidence: str) -> BeliefState:
        """Bayesian belief update given new evidence."""
        if proposition not in self.belief_system:
            # Create prior
            self.belief_system[proposition] = BeliefState(
                proposition=proposition,
                probability=0.5,
                confidence_interval=(0.3, 0.7)
            )

        belief = self.belief_system[proposition]

        # Use LLM to assess evidence strength
        prompt = f"""Current belief: "{proposition}" (probability: {belief.probability:.2f})
New evidence: {new_evidence}

How does this evidence change the probability?
Respond with: DIRECTION (UP/DOWN/NEUTRAL), MAGNITUDE (0.0-0.3), REASON"""

        try:
            response = await self.llm.generate_async(prompt, max_tokens=300, temperature=0.2)
            if response:
                direction = "UP" if "UP" in response.upper() else ("DOWN" if "DOWN" in response.upper() else "NEUTRAL")
                mag_match = re.search(r'([0-1]\.[0-9]+)', response)
                magnitude = float(mag_match.group(1)) if mag_match else 0.05

                if direction == "UP":
                    belief.probability = min(0.99, belief.probability + magnitude)
                    belief.evidence_for.append(new_evidence[:100])
                elif direction == "DOWN":
                    belief.probability = max(0.01, belief.probability - magnitude)
                    belief.evidence_against.append(new_evidence[:100])

                # Update CI
                half_ci = 0.15 * (1 - abs(belief.probability - 0.5) * 2)
                belief.confidence_interval = (
                    max(0, belief.probability - half_ci),
                    min(1, belief.probability + half_ci)
                )
                belief.last_updated = time.time()
                belief.update_count += 1

        except Exception as e:
            pass

        return belief

    async def strategic_plan(self, goal: str, constraints: list[str] = None, horizon_days: int = 30) -> dict:
        """
        Generate a strategic plan with probabilistic outcomes.
        Long-term thinking with obstacle anticipation.
        """
        constraints_text = '\n'.join(f"- {c}" for c in (constraints or []))

        prompt = f"""Create a strategic plan for: {goal}
Time horizon: {horizon_days} days
Constraints: {constraints_text or 'None specified'}

Provide:
1. OBJECTIVE: Clear measurable goal
2. KEY RESULTS: 3-5 measurable milestones
3. STRATEGIES: Top 3 approaches (with probability of success)
4. CRITICAL PATH: The most important steps in order
5. RISKS: Top 3 risks with mitigation
6. DECISION POINTS: When to pivot/adjust
7. SUCCESS METRICS: How to measure progress

Be specific and realistic."""

        try:
            plan_text = await self.llm.generate_async(prompt, max_tokens=2000, temperature=0.3)
        except Exception:
            plan_text = "Plan generation failed"

        # Adversarial review of the plan
        adversarial_prompt = f"""Review this plan and find its weaknesses:
{plan_text[:1000]}

What are the 3 most likely ways this plan will fail?"""

        try:
            risks = await self.llm.generate_async(adversarial_prompt, max_tokens=500, temperature=0.4)
        except Exception:
            risks = ""

        return {
            "goal": goal,
            "horizon_days": horizon_days,
            "plan": plan_text,
            "adversarial_risks": risks,
            "confidence": 0.72,
            "generated_at": time.time()
        }

    async def analyze_decision(self, decision: str, options: list[str]) -> dict:
        """
        Multi-criteria decision analysis with expected value calculation.
        Uses regret minimization + utility maximization.
        """
        options_text = '\n'.join(f"{i+1}. {opt}" for i, opt in enumerate(options))

        prompt = f"""Analyze this decision: {decision}

Options:
{options_text}

For each option provide:
- Expected value (1-10)
- Risk level (low/medium/high)
- Probability of success (0-1)
- Worst case outcome
- Best case outcome
- Regret if wrong (1-10)

Then recommend the best option with justification."""

        analysis = await self.llm.generate_async(prompt, max_tokens=1500, temperature=0.2)

        # Quick regret minimization
        minimax_prompt = f"""For decision: {decision}
Options: {options_text}
Which option minimizes maximum regret? (1 sentence)"""

        minimax = await self.llm.generate_async(minimax_prompt, max_tokens=200, temperature=0.2)

        return {
            "decision": decision,
            "options": options,
            "analysis": analysis,
            "minimax_regret": minimax,
            "framework": "expected_value + regret_minimization"
        }

    async def metacognitive_check(self) -> dict:
        """
        Self-assessment of current cognitive state.
        Am I thinking well? What should I do differently?
        """
        if len(self.thought_history) < 3:
            return {"status": "insufficient_history", "thoughts": len(self.thought_history)}

        recent_chains = self.thought_history[-10:]
        avg_quality = statistics.mean(c.quality_score for c in recent_chains)
        avg_confidence = statistics.mean(c.confidence for c in recent_chains)
        bias_rate = sum(len(c.biases_detected) > 0 for c in recent_chains) / len(recent_chains)

        assessment = f"""My recent thinking quality:
- Average quality score: {avg_quality:.2f}/1.0
- Average confidence: {avg_confidence:.2f}
- Bias detection rate: {bias_rate:.1%}
- Thoughts analyzed: {len(recent_chains)}

I should improve in areas where quality is low."""

        # Get LLM self-assessment
        improvement_prompt = f"""Based on this cognitive performance data:
- Quality: {avg_quality:.2f}/1.0 (target: 0.8)
- Confidence calibration: {avg_confidence:.2f}
- Bias incidence: {bias_rate:.1%}
- Total biases corrected: {self.cognitive_profile['biases_corrected']}

What are the top 2 improvements to make to reasoning quality?"""

        improvements = await self.llm.generate_async(improvement_prompt, max_tokens=400, temperature=0.3)

        metacog_entry = {
            "timestamp": time.time(),
            "avg_quality": avg_quality,
            "avg_confidence": avg_confidence,
            "bias_rate": bias_rate,
            "improvements": improvements
        }
        self.metacognitive_log.append(metacog_entry)

        return metacog_entry

    async def cross_domain_transfer(self, problem: str, source_domain: str) -> dict:
        """
        Apply insights from one domain to solve a problem in another.
        E.g., apply evolutionary biology principles to software architecture.
        """
        prompt = f"""Apply principles from {source_domain} to solve: {problem}

1. Key principles from {source_domain} that are relevant
2. How each principle maps to the problem
3. Novel solution derived from this analogy
4. What this domain transfer reveals that conventional thinking misses

Generate an innovative solution using {source_domain} as the lens:"""

        solution = await self.llm.generate_async(prompt, max_tokens=1200, temperature=0.6)

        return {
            "problem": problem,
            "source_domain": source_domain,
            "solution": solution,
            "method": "cross_domain_analogical_reasoning"
        }

    def get_cognitive_profile(self) -> dict:
        recent = self.thought_history[-20:] if self.thought_history else []
        return {
            **self.cognitive_profile,
            "belief_system_size": len(self.belief_system),
            "thought_history_size": len(self.thought_history),
            "metacognitive_checks": len(self.metacognitive_log),
            "recent_avg_quality": statistics.mean(c.quality_score for c in recent) if recent else 0.0,
            "recent_avg_confidence": statistics.mean(c.confidence for c in recent) if recent else 0.0,
        }

    def get_status(self) -> dict:
        return {
            "total_thoughts": self.cognitive_profile["total_thoughts"],
            "biases_detected": self.cognitive_profile["biases_detected"],
            "biases_corrected": self.cognitive_profile["biases_corrected"],
            "beliefs_held": len(self.belief_system),
            "metacognitive_checks": len(self.metacognitive_log)
        }

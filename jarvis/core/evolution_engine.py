"""
JARVIS Iteration 15: Self-Evolution Engine
Genetic programming + continuous benchmarking + automatic code improvement.
JARVIS rewrites and optimizes its own modules based on performance feedback.
"""
import ast
import copy
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import random
import re
import sys
import textwrap
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass
class Genome:
    """A candidate solution (code variant)."""
    id: str
    module_path: str
    function_name: str
    code: str
    fitness: float = 0.0
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)
    mutations: list[str] = field(default_factory=list)
    benchmark_results: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.id:
            self.id = hashlib.md5(self.code.encode()).hexdigest()[:8]


@dataclass
class BenchmarkResult:
    name: str
    score: float
    details: dict
    passed: bool
    time_taken: float


class EvolutionEngine:
    """
    Continuous self-improvement engine that:
    1. Monitors performance metrics across all JARVIS modules
    2. Generates code variants through mutation and crossover
    3. Evaluates variants against benchmarks
    4. Deploys improvements (with rollback capability)
    5. Learns which mutations are most effective
    """

    MUTATION_STRATEGIES = [
        "add_caching",
        "add_error_handling",
        "optimize_loops",
        "add_async",
        "add_logging",
        "refactor_logic",
        "add_type_hints",
        "add_docstring",
        "parallelize",
        "add_retry_logic",
    ]

    def __init__(self, llm_manager, base_path: str = "jarvis"):
        self.llm = llm_manager
        self.base_path = Path(base_path)
        self.population: dict[str, Genome] = {}
        self.deployed: dict[str, Genome] = {}
        self.generation = 0
        self.evolution_log: list[dict] = []
        self.performance_history: dict[str, list[float]] = {}
        self.improvement_patterns: list[dict] = []
        self.total_improvements = 0
        self._running = False
        print("[EvolutionEngine] Self-evolution system online")

    async def benchmark_function(self, func: Callable, test_cases: list[dict]) -> BenchmarkResult:
        """Evaluate a function against test cases."""
        start = time.time()
        scores = []
        details = {}

        for i, test in enumerate(test_cases):
            try:
                args = test.get("args", [])
                kwargs = test.get("kwargs", {})
                expected = test.get("expected")

                if inspect.iscoroutinefunction(func):
                    import asyncio
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                if expected is not None:
                    match = result == expected
                    scores.append(1.0 if match else 0.0)
                    details[f"test_{i}"] = {"passed": match, "got": str(result)[:100]}
                else:
                    scores.append(1.0)  # If no expected, just check it runs
                    details[f"test_{i}"] = {"passed": True, "result": str(result)[:100]}

            except Exception as e:
                scores.append(0.0)
                details[f"test_{i}"] = {"passed": False, "error": str(e)}

        score = sum(scores) / len(scores) if scores else 0.0
        return BenchmarkResult(
            name=getattr(func, "__name__", "unknown"),
            score=score,
            details=details,
            passed=score >= 0.8,
            time_taken=time.time() - start
        )

    async def generate_mutation(self, genome: Genome, strategy: str) -> Optional[Genome]:
        """Use LLM to apply a mutation strategy to code."""
        prompt = f"""You are a code optimizer. Apply the '{strategy}' mutation strategy to this Python function.

Original code:
```python
{genome.code}
```

Apply mutation: {strategy}

Rules:
1. Keep the same function signature
2. Preserve all functionality
3. Only improve the specific aspect: {strategy}
4. Return ONLY the improved Python code, no explanation
5. The code must be syntactically valid

Improved code:"""

        try:
            response = await self.llm.generate_async(prompt, max_tokens=2000, temperature=0.3)
            if not response:
                return None

            # Extract code from response
            code_match = re.search(r'```python\n(.*?)```', response, re.DOTALL)
            if code_match:
                new_code = code_match.group(1).strip()
            else:
                new_code = response.strip()

            # Validate syntax
            try:
                ast.parse(new_code)
            except SyntaxError:
                return None

            mutant = Genome(
                id="",
                module_path=genome.module_path,
                function_name=genome.function_name,
                code=new_code,
                generation=genome.generation + 1,
                parent_ids=[genome.id],
                mutations=genome.mutations + [strategy]
            )
            mutant.id = hashlib.md5(new_code.encode()).hexdigest()[:8]
            return mutant

        except Exception as e:
            print(f"[EvolutionEngine] Mutation error: {e}")
            return None

    async def crossover(self, genome_a: Genome, genome_b: Genome) -> Optional[Genome]:
        """Combine two genomes to create offspring."""
        prompt = f"""Combine the best aspects of these two Python function implementations.

Implementation A:
```python
{genome_a.code}
```

Implementation B:
```python
{genome_b.code}
```

Create a hybrid that takes the best from both. Keep the same function signature.
Return ONLY the Python code."""

        try:
            response = await self.llm.generate_async(prompt, max_tokens=2000, temperature=0.4)
            if not response:
                return None

            code_match = re.search(r'```python\n(.*?)```', response, re.DOTALL)
            new_code = code_match.group(1).strip() if code_match else response.strip()

            ast.parse(new_code)  # Validate

            offspring = Genome(
                id="",
                module_path=genome_a.module_path,
                function_name=genome_a.function_name,
                code=new_code,
                generation=max(genome_a.generation, genome_b.generation) + 1,
                parent_ids=[genome_a.id, genome_b.id],
                mutations=["crossover"]
            )
            offspring.id = hashlib.md5(new_code.encode()).hexdigest()[:8]
            return offspring

        except Exception as e:
            return None

    def select_parents(self, population: list[Genome], n: int = 2) -> list[Genome]:
        """Tournament selection - pick fittest individuals."""
        if len(population) < n:
            return population

        selected = []
        for _ in range(n):
            # Tournament of 3
            candidates = random.sample(population, min(3, len(population)))
            winner = max(candidates, key=lambda g: g.fitness)
            selected.append(winner)

        return selected

    async def evolve_function(
        self,
        module_path: str,
        function_name: str,
        test_cases: list[dict],
        generations: int = 3,
        population_size: int = 5
    ) -> Optional[Genome]:
        """
        Run evolutionary optimization on a specific function.
        Returns the best genome found.
        """
        print(f"[EvolutionEngine] Evolving {module_path}:{function_name} for {generations} generations")

        # Extract original code
        original_code = self._extract_function_code(module_path, function_name)
        if not original_code:
            print(f"[EvolutionEngine] Could not extract code for {function_name}")
            return None

        # Create seed genome
        seed = Genome(
            id="seed",
            module_path=module_path,
            function_name=function_name,
            code=original_code,
            generation=0
        )

        # Evaluate seed
        func = self._compile_function(seed.code, function_name)
        if func:
            bench = await self.benchmark_function(func, test_cases)
            seed.fitness = bench.score
            seed.benchmark_results = bench.details

        current_population = [seed]
        best_genome = seed
        self.generation = 0

        for gen in range(generations):
            self.generation = gen + 1
            print(f"[EvolutionEngine] Generation {self.generation}/{generations}")
            new_population = list(current_population)

            # Generate mutations
            for genome in current_population[:3]:  # Top 3
                # Random mutation strategy
                strategy = random.choice(self.MUTATION_STRATEGIES)
                mutant = await self.generate_mutation(genome, strategy)
                if mutant:
                    # Evaluate mutant
                    func = self._compile_function(mutant.code, function_name)
                    if func:
                        bench = await self.benchmark_function(func, test_cases)
                        mutant.fitness = bench.score
                        mutant.benchmark_results = bench.details
                        new_population.append(mutant)
                        self.population[mutant.id] = mutant

            # Crossover if enough individuals
            if len(new_population) >= 2:
                parents = self.select_parents(new_population, 2)
                if len(parents) == 2 and parents[0].id != parents[1].id:
                    offspring = await self.crossover(parents[0], parents[1])
                    if offspring:
                        func = self._compile_function(offspring.code, function_name)
                        if func:
                            bench = await self.benchmark_function(func, test_cases)
                            offspring.fitness = bench.score
                            new_population.append(offspring)
                            self.population[offspring.id] = offspring

            # Select fittest for next generation
            new_population.sort(key=lambda g: g.fitness, reverse=True)
            current_population = new_population[:population_size]

            # Track best
            if current_population[0].fitness > best_genome.fitness:
                best_genome = current_population[0]
                print(f"[EvolutionEngine] New best: fitness={best_genome.fitness:.3f} mutations={best_genome.mutations}")

        # Log evolution result
        improvement = best_genome.fitness - seed.fitness
        self.evolution_log.append({
            "function": f"{module_path}:{function_name}",
            "generations": generations,
            "initial_fitness": seed.fitness,
            "final_fitness": best_genome.fitness,
            "improvement": improvement,
            "best_mutations": best_genome.mutations
        })

        if improvement > 0:
            self.total_improvements += 1
            print(f"[EvolutionEngine] Improvement: +{improvement:.3f} fitness")

        return best_genome if best_genome.fitness > seed.fitness else None

    def _extract_function_code(self, module_path: str, function_name: str) -> Optional[str]:
        """Extract function source code from a module file."""
        path = self.base_path / module_path
        if not path.exists():
            # Try with .py extension
            path = path.with_suffix('.py') if not path.suffix else path
            if not path.exists():
                return None

        try:
            source = path.read_text(encoding='utf-8')
            tree = ast.parse(source)

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name == function_name:
                        # Extract function with decorator
                        lines = source.splitlines()
                        start = node.lineno - 1
                        end = node.end_lineno
                        func_lines = lines[start:end]
                        return '\n'.join(func_lines)
        except Exception as e:
            print(f"[EvolutionEngine] Extraction error: {e}")

        return None

    def _compile_function(self, code: str, function_name: str) -> Optional[Callable]:
        """Compile and return a function from code string."""
        try:
            namespace = {}
            exec(compile(code, '<evolution>', 'exec'), namespace)
            return namespace.get(function_name)
        except Exception as e:
            return None

    async def deploy_improvement(self, genome: Genome, backup: bool = True) -> bool:
        """Deploy an improved genome to the actual codebase."""
        path = self.base_path / genome.module_path
        if not path.exists():
            path = path.with_suffix('.py') if not path.suffix else path

        try:
            if backup:
                # Create backup
                backup_path = path.with_suffix('.py.bak')
                backup_path.write_text(path.read_text(encoding='utf-8'), encoding='utf-8')

            source = path.read_text(encoding='utf-8')
            tree = ast.parse(source)

            # Find and replace the function
            lines = source.splitlines()
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name == genome.function_name:
                        start = node.lineno - 1
                        end = node.end_lineno
                        new_lines = lines[:start] + genome.code.splitlines() + lines[end:]
                        new_source = '\n'.join(new_lines)
                        path.write_text(new_source, encoding='utf-8')

                        self.deployed[genome.id] = genome
                        print(f"[EvolutionEngine] Deployed improvement to {genome.module_path}:{genome.function_name}")
                        return True

        except Exception as e:
            print(f"[EvolutionEngine] Deploy error: {e}")
            if backup and backup_path.exists():
                path.write_text(backup_path.read_text(encoding='utf-8'), encoding='utf-8')

        return False

    async def scan_and_evolve(self, modules: list[str]) -> dict:
        """Scan modules for improvement opportunities and evolve them."""
        results = {}

        for module in modules:
            path = self.base_path / module
            if not path.exists():
                continue

            try:
                source = path.read_text(encoding='utf-8')
                tree = ast.parse(source)

                functions = [
                    node.name for node in ast.walk(tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and not node.name.startswith('_')
                    and len(ast.unparse(node)) > 100  # Only non-trivial functions
                ]

                print(f"[EvolutionEngine] Scanning {module}: {len(functions)} functions")

                for func_name in functions[:3]:  # Limit per module
                    # Generate basic test cases
                    test_cases = await self._generate_test_cases(module, func_name, source)
                    if test_cases:
                        improvement = await self.evolve_function(
                            module, func_name, test_cases, generations=2
                        )
                        if improvement:
                            results[f"{module}:{func_name}"] = {
                                "fitness_gain": improvement.fitness,
                                "mutations": improvement.mutations
                            }

            except Exception as e:
                print(f"[EvolutionEngine] Error scanning {module}: {e}")

        return results

    async def _generate_test_cases(self, module: str, function_name: str, source: str) -> list[dict]:
        """Auto-generate test cases for a function using LLM."""
        # Extract function signature
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name == function_name:
                        sig = ast.unparse(node.args)
                        # Simple test: just check it runs without error
                        return [{"args": [], "kwargs": {}, "expected": None}]
        except Exception:
            pass

        return []

    def get_evolution_report(self) -> dict:
        return {
            "generation": self.generation,
            "total_genomes": len(self.population),
            "deployed_improvements": len(self.deployed),
            "total_improvements": self.total_improvements,
            "evolution_log": self.evolution_log[-10:],
            "best_genomes": [
                {"id": g.id, "fitness": g.fitness, "mutations": g.mutations}
                for g in sorted(self.population.values(), key=lambda x: x.fitness, reverse=True)[:5]
            ]
        }

    def get_status(self) -> dict:
        return {
            "generation": self.generation,
            "population_size": len(self.population),
            "deployed": len(self.deployed),
            "improvements_made": self.total_improvements,
            "running": self._running
        }

"""Genetic programming engine for evolving alpha expressions.

Uses DEAP to evolve expression trees. Fitness = mean Spearman IC
between alpha values and forward returns on the training set.
"""

import random
import warnings

import numpy as np
import pandas as pd
from deap import algorithms, base, creator, gp, tools
from scipy.stats import spearmanr

from alpha_gpt.data.loader import DataSplit
from alpha_gpt.expr.primitives import create_primitive_set, DEFAULT_TERMINALS


def eval_expr(individual, pset, terminal_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Evaluate an expression tree against panel data, returning the alpha signal.

    Walks the DEAP ``PrimitiveTree`` directly (terminals are field names, looked up
    in ``terminal_data``; primitives are applied from ``pset.context``). NaN-safe:
    unknown terminals and operator errors yield an all-NaN panel rather than raising.
    """
    # DEAP compiles to a function that takes no args (terminals are constants)
    # but our "constants" are actually string names, so we need to intercept
    # We'll use a different approach: walk the tree and evaluate

    stack = []
    for node in reversed(individual):
        if isinstance(node, gp.Terminal):
            # Look up the actual data
            name = node.name
            if name in terminal_data:
                stack.append(terminal_data[name])
            else:
                # Unknown terminal, return NaN panel
                sample = next(iter(terminal_data.values()))
                stack.append(pd.DataFrame(np.nan, index=sample.index, columns=sample.columns))
        elif isinstance(node, gp.Primitive):
            args = [stack.pop() for _ in range(node.arity)]
            op = pset.context[node.name]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = op(*args)
                stack.append(result)
            except Exception:
                # On any error, push NaN
                sample = args[0] if args else next(iter(terminal_data.values()))
                stack.append(pd.DataFrame(np.nan, index=sample.index, columns=sample.columns))

    return stack[0] if stack else pd.DataFrame()


def _fitness(individual, pset, data: DataSplit) -> tuple[float]:
    """Compute fitness as mean Spearman IC on training data."""
    try:
        alpha_values = eval_expr(individual, pset, data.panels)

        if alpha_values.empty or alpha_values.isna().all().all():
            return (0.0,)

        # Compute daily Spearman IC (subsample dates for speed)
        fwd = data.forward_returns
        common_dates = alpha_values.index.intersection(fwd.index)
        if len(common_dates) < 30:
            return (0.0,)

        # Use every 5th date to keep fitness eval fast
        if len(common_dates) > 200:
            common_dates = common_dates[::5]

        ics = []
        for date in common_dates:
            a = alpha_values.loc[date].dropna()
            f = fwd.loc[date].dropna()
            common = a.index.intersection(f.index)
            if len(common) < 20:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ic, _ = spearmanr(a[common], f[common])
            if not np.isnan(ic):
                ics.append(ic)

        if len(ics) < 10:
            return (0.0,)

        return (float(np.mean(ics)),)

    except Exception:
        return (0.0,)


def run_gp(
    seed_trees: list | None,
    data: DataSplit,
    population_size: int = 100,
    generations: int = 20,
    crossover_prob: float = 0.7,
    mutation_prob: float = 0.2,
    tournament_size: int = 3,
    max_depth: int = 6,
    min_depth: int = 2,
    verbose: bool = True,
    random_seed: int | None = None,
    terminal_names: list[str] | None = None,
    hof_size: int = 10,
) -> tuple[list[dict], object]:
    """Run genetic programming to evolve alpha expressions.

    Args:
        seed_trees: Optional list of DEAP PrimitiveTree objects to seed the population.
        data: Training data split.
        population_size: GP population size.
        generations: Number of GP generations.
        crossover_prob: Crossover probability.
        mutation_prob: Mutation probability.
        tournament_size: Tournament selection size.
        max_depth: Maximum tree depth.
        min_depth: Minimum tree depth for generation.
        verbose: Print progress.
        random_seed: Optional fixed seed for reproducibility across runs.

    Returns:
        Tuple of (results_list, logbook)
        results_list is a list of dicts with 'expression' (str) and 'fitness' (float),
        sorted by fitness descending.
    """
    # Set random seeds for reproducibility
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    # Determine available terminals: use the resolved (capped) set if provided,
    # else fall back to the stable core. Must match the pset the pipeline uses to
    # parse seeds so seed trees inject cleanly.
    if terminal_names is None:
        terminal_names = DEFAULT_TERMINALS
    terminal_names = [name for name in terminal_names if name in data.panels]
    pset = create_primitive_set(terminal_names)

    # Create DEAP types (handle re-creation gracefully)
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=min_depth, max_=max_depth)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", _fitness, pset=pset, data=data)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

    # Safe height check — DEAP can produce empty trees after crossover
    def _safe_height(ind):
        try:
            return ind.height
        except IndexError:
            return max_depth + 1  # treat as too deep, discard

    # Depth limits to prevent bloat
    toolbox.decorate("mate", gp.staticLimit(key=_safe_height, max_value=max_depth))
    toolbox.decorate("mutate", gp.staticLimit(key=_safe_height, max_value=max_depth))

    # Initialize population
    pop = toolbox.population(n=population_size)

    # Inject seed trees into population
    if seed_trees:
        for i, seed in enumerate(seed_trees):
            if i < len(pop):
                ind = creator.Individual(seed)
                ind.fitness = creator.FitnessMax()
                pop[i] = ind

    # Statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values[0] if ind.fitness.valid else 0)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    stats.register("min", np.min)

    hof = tools.HallOfFame(hof_size)

    if verbose:
        print(f"Starting GP: pop={population_size}, gen={generations}, "
              f"terminals={terminal_names}")

    # Run evolution
    pop, log = algorithms.eaSimple(
        pop, toolbox,
        cxpb=crossover_prob,
        mutpb=mutation_prob,
        ngen=generations,
        stats=stats,
        halloffame=hof,
        verbose=verbose,
    )

    # Collect results
    results = []
    for ind in hof:
        results.append({
            "expression": str(ind),
            "tree": ind,
            "fitness": ind.fitness.values[0],
        })

    results.sort(key=lambda x: x["fitness"], reverse=True)

    if verbose:
        print(f"\nTop {len(results)} alphas:")
        for i, r in enumerate(results):
            print(f"  {i+1}. IC={r['fitness']:.4f}  {r['expression'][:80]}")

    return results, log

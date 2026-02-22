"""
=============================================================================
  Chapter 9: Population Methods
  Algorithms for Optimization — Kochenderfer & Wheeler (MIT Press, 2019)
=============================================================================

This file contains Python translations of all Julia algorithms from Chapter 9.
Each section is self-contained and runs a live demonstration at the end.

Sections:
  9.1  Initialization          — Uniform, Normal, Cauchy population sampling
  9.2  Genetic Algorithms      — Selection, Crossover, Mutation
  9.3  Differential Evolution
  9.4  Particle Swarm Optimization
  9.5  Firefly Algorithm
  9.6  Cuckoo Search
  9.7  Hybrid Methods          — Lamarckian vs Baldwinian (conceptual demo)
"""

import numpy as np
import random
from dataclasses import dataclass, field
from typing import Callable, List, Tuple
import warnings
warnings.filterwarnings("ignore")

# Set a global random seed for reproducibility (mirrors seed!(0) in Julia)
np.random.seed(0)
random.seed(0)

# =============================================================================
# SECTION 9.1 — INITIALIZATION
# =============================================================================
# Before running any population method, we need an initial spread of design
# points across the search space. Three strategies are provided below.

def rand_population_uniform(m: int, a: np.ndarray, b: np.ndarray) -> List[np.ndarray]:
    """
    Algorithm 9.1 — Uniform Population Sampling
    --------------------------------------------
    Samples m individuals uniformly at random within the hyperrectangle
    defined by lower bound vector `a` and upper bound vector `b`.

    Each coordinate x_i is drawn from U(a_i, b_i).

    Args:
        m: number of individuals (population size)
        a: lower bounds (array of length d)
        b: upper bounds (array of length d)

    Returns:
        List of m design points (numpy arrays)
    """
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    return [a + np.random.rand(len(a)) * (b - a) for _ in range(m)]


def rand_population_normal(m: int, mu: np.ndarray, sigma: np.ndarray) -> List[np.ndarray]:
    """
    Algorithm 9.2 — Normal Population Sampling
    -------------------------------------------
    Samples m individuals from a multivariate normal distribution.
    Useful when you have prior knowledge that the optimum lies near `mu`.

    Args:
        m:     number of individuals
        mu:    mean vector (array of length d)
        sigma: covariance matrix (d x d array)

    Returns:
        List of m design points
    """
    mu = np.array(mu, dtype=float)
    sigma = np.array(sigma, dtype=float)
    return [np.random.multivariate_normal(mu, sigma) for _ in range(m)]


def rand_population_cauchy(m: int, mu: np.ndarray, sigma: np.ndarray) -> List[np.ndarray]:
    """
    Algorithm 9.3 — Cauchy Population Sampling
    -------------------------------------------
    Samples m individuals from independent Cauchy distributions.
    The Cauchy distribution has HEAVY TAILS (undefined variance), meaning
    it can place individuals far from the center — useful for broad exploration.

    Args:
        m:     number of individuals
        mu:    location vector (analogous to mean)
        sigma: scale vector (analogous to std dev, but Cauchy has no std dev)

    Returns:
        List of m design points
    """
    mu = np.array(mu, dtype=float)
    sigma = np.array(sigma, dtype=float)
    return [
        np.array([np.random.standard_cauchy() * sigma[j] + mu[j]
                  for j in range(len(mu))])
        for _ in range(m)
    ]


def rand_population_binary(m: int, n: int) -> List[np.ndarray]:
    """
    Algorithm 9.5 — Binary Population Sampling
    -------------------------------------------
    Generates m random binary strings of length n.
    Used for genetic algorithms with binary-encoded chromosomes.

    Args:
        m: number of individuals
        n: length of each binary string

    Returns:
        List of m binary arrays (dtype bool)
    """
    return [np.random.randint(0, 2, size=n).astype(bool) for _ in range(m)]


# =============================================================================
# SECTION 9.2 — GENETIC ALGORITHMS
# =============================================================================
# Genetic algorithms mimic biological evolution:
#   1. SELECT fitter parents from the current population
#   2. CROSSOVER parent chromosomes to produce children
#   3. MUTATE children to introduce random variation
#   4. Repeat for k_max generations

# --- 9.2.3 SELECTION METHODS -------------------------------------------------

def select_truncation(y: np.ndarray, k: int) -> List[List[int]]:
    """
    Truncation Selection
    --------------------
    Keeps only the top-k fittest individuals (lowest objective values),
    then randomly pairs them as parents. Simple and fast, but can reduce
    diversity quickly.

    Args:
        y: array of objective function values for each individual
        k: number of top individuals to keep

    Returns:
        List of [parent1_idx, parent2_idx] pairs (one per child)
    """
    # Sort indices by ascending objective value (lower = fitter)
    sorted_idx = np.argsort(y)
    top_k = sorted_idx[:k]
    # Randomly pick 2 parents from the top-k for each child in the population
    return [list(np.random.choice(top_k, size=2, replace=True)) for _ in y]


def select_tournament(y: np.ndarray, k: int) -> List[List[int]]:
    """
    Tournament Selection
    --------------------
    For each parent slot: randomly pick k individuals, then choose the fittest.
    Balances selection pressure: larger k → more pressure toward best individuals.

    Args:
        y: array of objective function values
        k: tournament size

    Returns:
        List of [parent1_idx, parent2_idx] pairs
    """
    def get_parent():
        # Randomly select k candidates, return the index of the best one
        candidates = np.random.choice(len(y), size=k, replace=False)
        return candidates[np.argmin(y[candidates])]

    return [[get_parent(), get_parent()] for _ in y]


def select_roulette(y: np.ndarray) -> List[List[int]]:
    """
    Roulette Wheel Selection (Fitness-Proportionate Selection)
    ----------------------------------------------------------
    Each individual is selected with probability proportional to its fitness.
    Fitness = max(y) - y(i), so the worst individual has zero chance.

    Args:
        y: array of objective function values

    Returns:
        List of [parent1_idx, parent2_idx] pairs
    """
    # Invert so lower objective = higher fitness
    fitness = np.max(y) - y
    total = fitness.sum()
    if total == 0:
        # All individuals are equal — pick uniformly
        probs = np.ones(len(y)) / len(y)
    else:
        probs = fitness / total
    return [list(np.random.choice(len(y), size=2, replace=True, p=probs)) for _ in y]


# --- 9.2.4 CROSSOVER METHODS -------------------------------------------------

def crossover_single_point(a, b):
    """
    Single-Point Crossover
    ----------------------
    Picks a random cut point i. The child takes genes [0:i] from parent a
    and genes [i:] from parent b.

    Works for both binary and real-valued chromosomes.
    """
    i = np.random.randint(1, len(a))
    if isinstance(a, np.ndarray) and a.dtype == bool:
        return np.concatenate([a[:i], b[i:]])
    return list(a[:i]) + list(b[i:])


def crossover_two_point(a, b):
    """
    Two-Point Crossover
    -------------------
    Picks two random cut points i < j.
    Child = a[0:i] + b[i:j] + a[j:]
    Allows more recombination than single-point.
    """
    n = len(a)
    i, j = sorted(np.random.choice(n, size=2, replace=False))
    if isinstance(a, np.ndarray) and a.dtype == bool:
        return np.concatenate([a[:i], b[i:j], a[j:]])
    return list(a[:i]) + list(b[i:j]) + list(a[j:])


def crossover_uniform(a, b):
    """
    Uniform Crossover
    -----------------
    Each gene is independently taken from parent a or b with 50/50 probability.
    Maximum recombination — child is a random mix of both parents.
    """
    child = a.copy() if isinstance(a, np.ndarray) else list(a)
    for i in range(len(a)):
        if np.random.rand() < 0.5:
            child[i] = b[i]
    return child


def crossover_interpolation(a: np.ndarray, b: np.ndarray, lam: float = 0.5) -> np.ndarray:
    """
    Interpolation Crossover (Real-Valued Only)
    ------------------------------------------
    Produces a child that is a weighted average of the two parents:
        child = (1 - λ) * a + λ * b
    With λ=0.5, the child is the midpoint of the two parents.

    Args:
        a, b: parent chromosomes (real-valued numpy arrays)
        lam:  interpolation weight in [0, 1]
    """
    return (1 - lam) * np.array(a) + lam * np.array(b)


# --- 9.2.5 MUTATION METHODS --------------------------------------------------

def mutate_bitwise(child: np.ndarray, lam: float) -> np.ndarray:
    """
    Bitwise Mutation (Binary Chromosomes)
    --------------------------------------
    Each bit is independently flipped with probability λ.
    Typical setting: λ = 1/m where m is chromosome length,
    so on average ONE bit is flipped per child.

    Args:
        child: binary chromosome
        lam:   mutation probability per bit
    """
    return np.array([not v if np.random.rand() < lam else v for v in child])


def mutate_gaussian(child: np.ndarray, sigma: float) -> np.ndarray:
    """
    Gaussian Mutation (Real-Valued Chromosomes)
    --------------------------------------------
    Adds zero-mean Gaussian noise to each gene.
    Allows fine-grained local exploration around current values.

    Args:
        child: real-valued chromosome
        sigma: standard deviation of the noise
    """
    return np.array(child) + np.random.randn(len(child)) * sigma


# --- 9.2 MAIN GENETIC ALGORITHM ----------------------------------------------

def genetic_algorithm(
    f: Callable,
    population: List,
    k_max: int,
    select_fn: Callable,
    crossover_fn: Callable,
    mutate_fn: Callable
):
    """
    Algorithm 9.4 — Genetic Algorithm
    -----------------------------------
    Main loop for the genetic algorithm. At each generation:
      1. Evaluate fitness of all individuals
      2. Select parents using the provided selection strategy
      3. Create children via crossover of selected parents
      4. Mutate each child
      5. Replace the old population with the new one

    Args:
        f:            objective function to MINIMIZE
        population:   list of initial design points (chromosomes)
        k_max:        number of generations to run
        select_fn:    function(y) -> list of [p1_idx, p2_idx] pairs
        crossover_fn: function(a, b) -> child chromosome
        mutate_fn:    function(child) -> mutated child

    Returns:
        The best individual found across all generations
    """
    for generation in range(k_max):
        # Evaluate objective function for all individuals
        y = np.array([f(ind) for ind in population])

        # Select parent pairs
        parent_pairs = select_fn(y)

        # Produce children via crossover then mutation
        children = [
            mutate_fn(crossover_fn(population[p[0]], population[p[1]]))
            for p in parent_pairs
        ]
        population = children

    # Return the individual with the lowest objective value
    y_final = np.array([f(ind) for ind in population])
    return population[np.argmin(y_final)]


# =============================================================================
# SECTION 9.3 — DIFFERENTIAL EVOLUTION
# =============================================================================
# Differential Evolution (DE) improves each individual by blending it with
# three other randomly chosen individuals. It is simple, effective, and
# requires few hyperparameters.

def differential_evolution(
    f: Callable,
    population: List[np.ndarray],
    k_max: int,
    p: float = 0.5,
    w: float = 1.0
) -> np.ndarray:
    """
    Algorithm 9.10 — Differential Evolution
    -----------------------------------------
    For each individual x in the population:
      1. Pick 3 random distinct others: a, b, c
      2. Create trial vector: z = a + w * (b - c)
      3. Build candidate x' using binary crossover between x and z
      4. If f(x') < f(x), replace x with x'

    Args:
        f:       objective function to MINIMIZE
        population: list of real-valued design points
        k_max:   number of iterations (passes over entire population)
        p:       crossover probability — higher p → more mixing
        w:       differential weight (scale of perturbation), typically 0.4–1.0

    Returns:
        Best design point found
    """
    population = [np.array(x, dtype=float) for x in population]
    m = len(population)
    n = len(population[0])

    for _ in range(k_max):
        for k in range(m):
            x = population[k]

            # Select 3 distinct individuals different from current index k
            candidates = [j for j in range(m) if j != k]
            a, b, c = [population[i] for i in random.sample(candidates, 3)]

            # Mutation: create trial vector
            z = a + w * (b - c)

            # Crossover: guarantee at least one dimension comes from z
            j_rand = np.random.randint(0, n)
            x_prime = np.array([
                z[i] if (i == j_rand or np.random.rand() < p) else x[i]
                for i in range(n)
            ])

            # Greedy selection: keep whichever is better
            if f(x_prime) < f(x):
                population[k] = x_prime

    y_all = np.array([f(x) for x in population])
    return population[np.argmin(y_all)]


# =============================================================================
# SECTION 9.4 — PARTICLE SWARM OPTIMIZATION
# =============================================================================
# PSO models each individual as a "particle" flying through the search space.
# Particles have velocity and are attracted toward:
#   - their own personal best position (cognitive component)
#   - the global best position found by any particle (social component)

@dataclass
class Particle:
    """
    Algorithm 9.11 — Particle
    -------------------------
    Represents a single particle in PSO.

    Attributes:
        x:      current position in design space
        v:      current velocity vector
        x_best: best position this particle has personally visited
    """
    x: np.ndarray
    v: np.ndarray
    x_best: np.ndarray


def particle_swarm_optimization(
    f: Callable,
    population: List[Particle],
    k_max: int,
    w: float = 1.0,
    c1: float = 1.0,
    c2: float = 1.0
) -> List[Particle]:
    """
    Algorithm 9.12 — Particle Swarm Optimization
    ----------------------------------------------
    At each iteration, every particle updates its velocity and position:

        v ← w*v + c1*r1*(x_best - x) + c2*r2*(global_best - x)
        x ← x + v

    where r1, r2 ~ U(0,1) are random vectors (per dimension).

    Args:
        f:          objective function to MINIMIZE
        population: list of Particle objects
        k_max:      number of iterations
        w:          inertia weight (controls momentum; can decay over time)
        c1:         cognitive coefficient (attraction to personal best)
        c2:         social coefficient (attraction to global best)

    Returns:
        Updated list of particles (inspect .x_best for results)
    """
    # Initialize global best from the starting population
    x_best_global = population[0].x_best.copy()
    y_best_global = np.inf

    for p in population:
        y = f(p.x)
        if y < y_best_global:
            x_best_global = p.x.copy()
            y_best_global = y

    for _ in range(k_max):
        for p in population:
            n = len(p.x)
            r1 = np.random.rand(n)
            r2 = np.random.rand(n)

            # Update position
            p.x = p.x + p.v

            # Update velocity: inertia + cognitive pull + social pull
            p.v = (w * p.v
                   + c1 * r1 * (p.x_best - p.x)
                   + c2 * r2 * (x_best_global - p.x))

            y = f(p.x)

            # Update global best
            if y < y_best_global:
                x_best_global = p.x.copy()
                y_best_global = y

            # Update personal best
            if y < f(p.x_best):
                p.x_best = p.x.copy()

    return population


# =============================================================================
# SECTION 9.5 — FIREFLY ALGORITHM
# =============================================================================
# Inspired by firefly bioluminescence. Brighter (fitter) fireflies attract
# dimmer ones. Attraction decreases with distance, modeled by a brightness
# function I(r). A random walk component ensures exploration.

def firefly(
    f: Callable,
    population: List[np.ndarray],
    k_max: int,
    beta: float = 1.0,
    alpha: float = 0.1,
    brightness: Callable = lambda r: np.exp(-r**2)
) -> np.ndarray:
    """
    Algorithm 9.13 — Firefly Algorithm
    ------------------------------------
    At each iteration, for every pair of fireflies (a, b):
      - If f(b) < f(a), i.e., b is brighter (fitter):
        Move a toward b:
          a ← a + β * I(||b - a||) * (b - a) + α * ε
        where ε ~ N(0, I) is Gaussian noise for random exploration.

    Args:
        f:          objective function to MINIMIZE
        population: list of design points (firefly positions)
        k_max:      number of iterations
        beta:       source intensity (controls attraction strength)
        alpha:      random walk step size (exploration noise)
        brightness: intensity function I(r) — decreases with distance r.
                    Default: Gaussian I(r) = exp(-r²), avoids singularity at 0.

    Returns:
        Best design point found
    """
    population = [np.array(x, dtype=float) for x in population]
    d = len(population[0])

    for _ in range(k_max):
        for i in range(len(population)):
            for j in range(len(population)):
                if f(population[j]) < f(population[i]):
                    # j is brighter → move i toward j
                    r = np.linalg.norm(population[j] - population[i])
                    noise = np.random.randn(d)
                    population[i] = (population[i]
                                     + beta * brightness(r) * (population[j] - population[i])
                                     + alpha * noise)

    y_all = np.array([f(x) for x in population])
    return population[np.argmin(y_all)]


# =============================================================================
# SECTION 9.6 — CUCKOO SEARCH
# =============================================================================
# Inspired by cuckoo brood parasitism. Each "nest" is a design point.
# New nests are explored via Lévy-like flights (here, Cauchy steps).
# A fraction of the worst nests are abandoned each generation.

@dataclass
class Nest:
    """
    Represents a single nest in Cuckoo Search.

    Attributes:
        x: position (design point)
        y: objective value f(x)
    """
    x: np.ndarray
    y: float


def cauchy_step(scale: float = 1.0) -> float:
    """Draws a single Cauchy-distributed random step."""
    return np.random.standard_cauchy() * scale


def cuckoo_search(
    f: Callable,
    population: List[Nest],
    k_max: int,
    p_a: float = 0.1,
    cauchy_scale: float = 1.0
) -> List[Nest]:
    """
    Algorithm 9.14 — Cuckoo Search
    --------------------------------
    At each iteration:
      1. A random cuckoo lays an egg near a random nest (Cauchy flight).
         If the new egg is better than the chosen nest, it replaces it.
      2. The worst `a = floor(m * p_a)` nests are abandoned and replaced
         by new nests generated via Cauchy flights from surviving nests.

    The Cauchy distribution is used for flights because its heavy tail mimics
    animal foraging behavior and allows large exploratory jumps.

    Args:
        f:             objective function to MINIMIZE
        population:    list of Nest objects
        k_max:         number of iterations
        p_a:           fraction of nests to abandon each generation (0–1)
        cauchy_scale:  scale parameter for the Cauchy flight distribution

    Returns:
        Final population of nests (sorted best to worst)
    """
    m = len(population)
    n = len(population[0].x)
    a = max(1, round(m * p_a))  # number of nests to abandon

    for _ in range(k_max):
        # --- Step 1: Cuckoo lays egg in a random nest ---
        i = np.random.randint(0, m)  # nest to potentially replace
        j = np.random.randint(0, m)  # source nest for the flight

        # Lévy-like flight via Cauchy steps
        step = np.array([cauchy_step(cauchy_scale) for _ in range(n)])
        x_new = population[j].x + step
        y_new = f(x_new)

        if y_new < population[i].y:
            population[i] = Nest(x=x_new, y=y_new)

        # --- Step 2: Abandon worst nests ---
        # Sort: best nests first (lowest y)
        population.sort(key=lambda nest: nest.y)

        # Replace the worst `a` nests with Cauchy flights from surviving nests
        for bad_idx in range(m - a, m):
            src_idx = np.random.randint(0, m - a)  # fly from a surviving nest
            step = np.array([cauchy_step(cauchy_scale) for _ in range(n)])
            x_new = population[src_idx].x + step
            population[bad_idx] = Nest(x=x_new, y=f(x_new))

    population.sort(key=lambda nest: nest.y)
    return population


# =============================================================================
# SECTION 9.7 — HYBRID METHODS (Conceptual Demo)
# =============================================================================
# Population methods explore broadly but converge slowly.
# Descent methods converge quickly but get stuck in local minima.
# Hybrid methods combine both: after crossover/mutation, apply a local search.

def local_search_step(f: Callable, x: np.ndarray, step_size: float = 0.01, n_steps: int = 10) -> np.ndarray:
    """
    A simple gradient-free local search using coordinate descent.
    For each dimension, tries a small perturbation and keeps it if better.
    """
    x = x.copy()
    for _ in range(n_steps):
        for d in range(len(x)):
            for delta in [step_size, -step_size]:
                x_trial = x.copy()
                x_trial[d] += delta
                if f(x_trial) < f(x):
                    x = x_trial
    return x


def lamarckian_step(f: Callable, population: List[np.ndarray], step_size: float = 0.01) -> List[np.ndarray]:
    """
    Lamarckian Learning
    --------------------
    Apply local search to each individual and REPLACE the individual
    with its locally optimized version. The improved position and its
    objective value are what the next generation "sees".

    Risk: can cause premature convergence to a local optimum.
    """
    return [local_search_step(f, x, step_size) for x in population]


def baldwinian_step(f: Callable, population: List[np.ndarray], step_size: float = 0.01) -> Tuple[List[np.ndarray], List[float]]:
    """
    Baldwinian Learning
    --------------------
    Apply local search to each individual to estimate its potential fitness,
    but DO NOT replace the individual's actual position. The selection step
    uses the optimized fitness values, but individuals remain in their original
    positions. This helps avoid premature convergence while still guiding
    selection toward promising regions.

    Returns:
        Original population (unchanged positions) + estimated fitness values
    """
    perceived_fitness = [f(local_search_step(f, x, step_size)) for x in population]
    return population, perceived_fitness  # positions unchanged, only fitness updated


# =============================================================================
# HELPER: TEST FUNCTIONS
# =============================================================================

def sphere(x: np.ndarray) -> float:
    """Sphere function: f(x) = ||x||². Global minimum at origin = 0."""
    return float(np.dot(x, x))


def ackley(x: np.ndarray) -> float:
    """
    Ackley function: highly multimodal, global minimum at origin = 0.
    Good for testing global optimization methods.
    """
    n = len(x)
    a, b, c = 20, 0.2, 2 * np.pi
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    return (-a * np.exp(-b * np.sqrt(sum1 / n))
            - np.exp(sum2 / n)
            + a + np.e)


def two_hump(x: np.ndarray) -> float:
    """
    Two-hump function for Lamarckian/Baldwinian demo.
    f(x) = -exp(-x²) - 2*exp(-(x-3)²)
    Has a local minimum near x=0 and global minimum near x=3.
    """
    return -np.exp(-x[0]**2) - 2 * np.exp(-(x[0] - 3)**2)


# =============================================================================
# MAIN — RUN ALL DEMONSTRATIONS
# =============================================================================

def separator(title: str):
    """Prints a formatted section separator."""
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def run_all():
    np.random.seed(42)
    random.seed(42)

    # -------------------------------------------------------------------------
    separator("9.1 INITIALIZATION — Population Sampling")
    # -------------------------------------------------------------------------

    a_bounds = np.array([-2.0, -2.0])
    b_bounds = np.array([ 2.0,  2.0])

    pop_uniform = rand_population_uniform(5, a_bounds, b_bounds)
    print("\n[Uniform] 5 design points sampled from [-2,2]²:")
    for pt in pop_uniform:
        print(f"  {pt.round(4)}")

    pop_normal = rand_population_normal(5, mu=np.zeros(2), sigma=np.eye(2))
    print("\n[Normal]  5 design points sampled from N(0, I):")
    for pt in pop_normal:
        print(f"  {pt.round(4)}")

    pop_cauchy = rand_population_cauchy(5, mu=np.zeros(2), sigma=np.ones(2))
    print("\n[Cauchy]  5 design points sampled with Cauchy(0,1):")
    for pt in pop_cauchy:
        print(f"  {pt.round(4)}")

    pop_binary = rand_population_binary(3, n=8)
    print("\n[Binary]  3 binary chromosomes of length 8:")
    for ch in pop_binary:
        print(f"  {ch.astype(int)}")

    # -------------------------------------------------------------------------
    separator("9.2.3 SELECTION METHODS")
    # -------------------------------------------------------------------------

    y_demo = np.array([3.5, 1.2, 4.8, 0.7, 2.1])
    print(f"\nObjective values: {y_demo}")
    print(f"Best individual:  index {np.argmin(y_demo)} (y={y_demo.min()})")

    pairs_trunc = select_truncation(y_demo, k=2)
    print(f"\n[Truncation k=2] Selected parent pairs: {pairs_trunc[:3]}...")

    pairs_tourn = select_tournament(y_demo, k=2)
    print(f"[Tournament k=2] Selected parent pairs: {pairs_tourn[:3]}...")

    pairs_roule = select_roulette(y_demo)
    print(f"[Roulette]       Selected parent pairs: {pairs_roule[:3]}...")

    # -------------------------------------------------------------------------
    separator("9.2.4 CROSSOVER METHODS")
    # -------------------------------------------------------------------------

    a_chr = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=bool)
    b_chr = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=bool)

    print(f"\nParent A: {a_chr.astype(int)}")
    print(f"Parent B: {b_chr.astype(int)}")
    print(f"[Single-Point ] Child: {crossover_single_point(a_chr, b_chr).astype(int)}")
    print(f"[Two-Point    ] Child: {crossover_two_point(a_chr, b_chr).astype(int)}")
    print(f"[Uniform      ] Child: {crossover_uniform(a_chr, b_chr).astype(int)}")

    # Real-valued interpolation crossover
    a_real = np.array([1.0, 2.0, 3.0])
    b_real = np.array([4.0, 5.0, 6.0])
    print(f"\n[Interpolation λ=0.5] a={a_real}, b={b_real}")
    print(f"  Child: {crossover_interpolation(a_real, b_real, lam=0.5)}")

    # -------------------------------------------------------------------------
    separator("9.2.5 MUTATION METHODS")
    # -------------------------------------------------------------------------

    child_bin  = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=bool)
    child_real = np.array([1.5, -0.3, 2.7])

    print(f"\n[Bitwise  λ=0.2] Before: {child_bin.astype(int)}")
    print(f"                 After:  {mutate_bitwise(child_bin, lam=0.2).astype(int)}")

    print(f"\n[Gaussian σ=0.1] Before: {child_real}")
    print(f"                 After:  {mutate_gaussian(child_real, sigma=0.1).round(4)}")

    # -------------------------------------------------------------------------
    separator("9.2 GENETIC ALGORITHM — Minimize ||x|| (Sphere)")
    # -------------------------------------------------------------------------
    # Mirrors Example 9.1 from the book.
    print("\nObjective: minimize f(x) = ||x|| (norm), x ∈ [-3,3]²")
    print("Strategy: TruncationSelection(k=10), SinglePointCrossover, GaussianMutation(σ=0.5)")

    m_ga      = 100
    k_max_ga  = 10
    top_k     = 10
    sigma_mut = 0.5
    a_ga      = np.array([-3.0, -3.0])
    b_ga      = np.array([ 3.0,  3.0])

    pop_ga = rand_population_uniform(m_ga, a_ga, b_ga)

    result_ga = genetic_algorithm(
        f           = lambda x: np.linalg.norm(x),
        population  = pop_ga,
        k_max       = k_max_ga,
        select_fn   = lambda y: select_truncation(y, k=top_k),
        crossover_fn= crossover_single_point,
        mutate_fn   = lambda c: mutate_gaussian(np.array(c, dtype=float), sigma_mut)
    )
    print(f"\nBest x found:  {result_ga.round(6)}")
    print(f"f(x) = ||x||:  {np.linalg.norm(result_ga):.6f}  (optimum = 0)")

    # -------------------------------------------------------------------------
    separator("9.3 DIFFERENTIAL EVOLUTION — Minimize Ackley function")
    # -------------------------------------------------------------------------
    print("\nObjective: minimize Ackley(x), x ∈ [-5,5]²")
    print("Parameters: p=0.5 (crossover prob), w=0.8 (differential weight)")

    pop_de = rand_population_uniform(30, np.array([-5.0, -5.0]), np.array([5.0, 5.0]))

    best_de = differential_evolution(
        f=ackley, population=pop_de, k_max=200, p=0.5, w=0.8
    )
    print(f"\nBest x found:  {best_de.round(6)}")
    print(f"Ackley(x):     {ackley(best_de):.6f}  (optimum = 0)")

    # -------------------------------------------------------------------------
    separator("9.4 PARTICLE SWARM OPTIMIZATION — Minimize Ackley function")
    # -------------------------------------------------------------------------
    print("\nObjective: minimize Ackley(x), x ∈ [-5,5]²")
    print("Parameters: w=0.5 (inertia), c1=1.5 (cognitive), c2=1.5 (social)")

    def make_particle(x):
        """Helper: create a Particle with zero initial velocity."""
        return Particle(x=x.copy(), v=np.zeros_like(x), x_best=x.copy())

    pop_pso_init = rand_population_uniform(30, np.array([-5.0, -5.0]), np.array([5.0, 5.0]))
    pop_pso = [make_particle(x) for x in pop_pso_init]

    pop_pso_result = particle_swarm_optimization(
        f=ackley, population=pop_pso, k_max=200, w=0.5, c1=1.5, c2=1.5
    )

    best_pso = min(pop_pso_result, key=lambda p: ackley(p.x_best))
    print(f"\nBest x found:  {best_pso.x_best.round(6)}")
    print(f"Ackley(x):     {ackley(best_pso.x_best):.6f}  (optimum = 0)")

    # -------------------------------------------------------------------------
    separator("9.5 FIREFLY ALGORITHM — Minimize Sphere function")
    # -------------------------------------------------------------------------
    print("\nObjective: minimize Sphere(x) = ||x||², x ∈ [-5,5]²")
    print("Parameters: β=1.0, α=0.2, brightness=exp(-r²)")
    print("Note: O(m²) per iteration — using small population for speed.")

    pop_ff = rand_population_uniform(20, np.array([-5.0, -5.0]), np.array([5.0, 5.0]))

    best_ff = firefly(
        f=sphere, population=pop_ff, k_max=50,
        beta=1.0, alpha=0.2,
        brightness=lambda r: np.exp(-r**2)
    )
    print(f"\nBest x found:  {best_ff.round(6)}")
    print(f"Sphere(x):     {sphere(best_ff):.6f}  (optimum = 0)")

    # -------------------------------------------------------------------------
    separator("9.6 CUCKOO SEARCH — Minimize Sphere function")
    # -------------------------------------------------------------------------
    print("\nObjective: minimize Sphere(x) = ||x||², x ∈ [-5,5]²")
    print("Parameters: p_a=0.25 (abandon rate), Cauchy scale=0.5")

    pop_cs_init = rand_population_uniform(20, np.array([-5.0, -5.0]), np.array([5.0, 5.0]))
    pop_cs = [Nest(x=x, y=sphere(x)) for x in pop_cs_init]

    pop_cs_result = cuckoo_search(
        f=sphere, population=pop_cs, k_max=200, p_a=0.25, cauchy_scale=0.5
    )

    best_cs = pop_cs_result[0]  # sorted best-first
    print(f"\nBest x found:  {best_cs.x.round(6)}")
    print(f"Sphere(x):     {best_cs.y:.6f}  (optimum = 0)")

    # -------------------------------------------------------------------------
    separator("9.7 HYBRID METHODS — Lamarckian vs Baldwinian")
    # -------------------------------------------------------------------------
    # f(x) = -exp(-x²) - 2*exp(-(x-3)²)
    # Has a local min near x=0, global min near x=3.
    # Population initialized near x=0 will trap a Lamarckian search.

    print("\nObjective: f(x) = -exp(-x²) - 2*exp(-(x-3)²)")
    print("Local min ≈ x=0,  Global min ≈ x=3")
    print("Population initialized near x=0")

    pop_hybrid = [np.array([np.random.uniform(-0.5, 0.5)]) for _ in range(6)]
    print(f"\nInitial population: {[round(x[0], 3) for x in pop_hybrid]}")

    # Lamarckian: individuals are replaced by their locally improved versions
    pop_lamarck = lamarckian_step(two_hump, pop_hybrid.copy(), step_size=0.05)
    print(f"\n[Lamarckian] After local search — positions REPLACED:")
    print(f"  Positions: {[round(x[0], 3) for x in pop_lamarck]}")
    print(f"  Values:    {[round(two_hump(x), 4) for x in pop_lamarck]}")
    print(f"  → All converged to local min near 0. Global min at 3 is MISSED.")

    # Baldwinian: positions unchanged, only perceived fitness updated
    pop_baldwin, perceived = baldwinian_step(two_hump, pop_hybrid.copy(), step_size=0.05)
    print(f"\n[Baldwinian] After local search — positions UNCHANGED:")
    print(f"  Positions:         {[round(x[0], 3) for x in pop_baldwin]}")
    print(f"  Perceived fitness: {[round(v, 4) for v in perceived]}")
    print(f"  → Positions kept. Selection guided by optimized fitness values.")
    print(f"  → Future crossover can still reach the global min at x≈3.")

    # -------------------------------------------------------------------------
    separator("SUMMARY OF RESULTS")
    # -------------------------------------------------------------------------
    print("""
  Algorithm                | Problem       | Best f(x) found
  -------------------------|---------------|------------------
  Genetic Algorithm        | Sphere (2D)   | {ga:.6f}
  Differential Evolution   | Ackley (2D)   | {de:.6f}
  Particle Swarm Optim.    | Ackley (2D)   | {pso:.6f}
  Firefly Algorithm        | Sphere (2D)   | {ff:.6f}
  Cuckoo Search            | Sphere (2D)   | {cs:.6f}
  
  All optima = 0 for Sphere/Ackley. Closer to 0 = better.
    """.format(
        ga  = float(np.linalg.norm(result_ga)),
        de  = float(ackley(best_de)),
        pso = float(ackley(best_pso.x_best)),
        ff  = float(sphere(best_ff)),
        cs  = float(best_cs.y),
    ))


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_all()

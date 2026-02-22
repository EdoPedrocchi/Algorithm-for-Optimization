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

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch

# ── Academic style palette ───────────────────────────────────────────────────
import matplotlib as mpl
mpl.rcParams.update({
    "font.family":       "serif",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#e0e0e0",
    "grid.linewidth":    0.5,
    "legend.framealpha": 0.9,
    "legend.fontsize":   8,
    "figure.dpi":        120,
})

# Neutral academic colors (colorblind-friendly)
DARK   = "white"      # figure background
MID    = "white"      # axes background
ACCENT = "#333333"    # spines
GOLD   = "#d62728"    # red
TEAL   = "#1f77b4"    # blue
GREEN  = "#2ca02c"    # green
ORANGE = "#ff7f0e"    # orange
WHITE  = "black"      # text (inverted for compatibility)

def apply_dark_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("white")
    ax.tick_params(colors="black", labelsize=8)
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    ax.title.set_color("black")
    for spine in ax.spines.values():
        spine.set_edgecolor("#aaaaaa")
    if title:  ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    if xlabel: ax.set_xlabel(xlabel, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9)

def make_fig(nrows, ncols, figsize, title):
    fig = plt.figure(figsize=figsize, facecolor="white")
    fig.suptitle(title, color="black", fontsize=12, fontweight="bold", y=0.98)
    return fig

# ── Contour helper ────────────────────────────────────────────────────────────
def contour_bg(ax, f, xlim, ylim, res=200, cmap="Blues", levels=30):
    xs = np.linspace(*xlim, res)
    ys = np.linspace(*ylim, res)
    X, Y = np.meshgrid(xs, ys)
    Z = np.array([[f(np.array([x, y])) for x in xs] for y in ys])
    ax.contourf(X, Y, Z, levels=levels, cmap=cmap, alpha=0.75)
    ax.contour(X, Y, Z, levels=8, colors="white", linewidths=0.3, alpha=0.5)

def _show_plot(fig):
    """Display a figure inline in Jupyter/Colab, or save/show in terminal."""
    try:
        from IPython.display import display as ipy_display
        from io import BytesIO
        from IPython.display import Image as IPyImage
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
        buf.seek(0)
        ipy_display(IPyImage(buf.read()))
        plt.close(fig)
    except ImportError:
        plt.show()




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
# MAIN — RUN ALL DEMONSTRATIONS + PLOTS
# =============================================================================

def separator(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


# ── Instrumented versions of algorithms (record history for plotting) ─────────

def genetic_algorithm_history(f, population, k_max, select_fn, crossover_fn, mutate_fn):
    """Returns list-of-populations (one per generation) for plotting."""
    history = [[x.copy() for x in population]]
    for _ in range(k_max):
        y = np.array([f(ind) for ind in population])
        parent_pairs = select_fn(y)
        children = [mutate_fn(crossover_fn(population[p[0]], population[p[1]]))
                    for p in parent_pairs]
        population = children
        history.append([x.copy() for x in population])
    return history


def differential_evolution_history(f, population, k_max, p=0.5, w=1.0):
    """Returns (history_of_populations, best_per_iter)."""
    population = [np.array(x, dtype=float) for x in population]
    m, n = len(population), len(population[0])
    history = [[x.copy() for x in population]]
    best_vals = [min(f(x) for x in population)]
    for _ in range(k_max):
        for k in range(m):
            x = population[k]
            candidates = [j for j in range(m) if j != k]
            a, b, c = [population[i] for i in random.sample(candidates, 3)]
            z = a + w * (b - c)
            j_rand = np.random.randint(0, n)
            x_prime = np.array([z[i] if (i == j_rand or np.random.rand() < p) else x[i]
                                 for i in range(n)])
            if f(x_prime) < f(x):
                population[k] = x_prime
        history.append([x.copy() for x in population])
        best_vals.append(min(f(x) for x in population))
    return history, best_vals


def pso_history(f, population, k_max, w=0.5, c1=1.5, c2=1.5):
    """Returns (history_of_positions, best_per_iter)."""
    import copy
    population = copy.deepcopy(population)
    x_best_global = population[0].x_best.copy()
    y_best_global = np.inf
    for p in population:
        y = f(p.x)
        if y < y_best_global:
            x_best_global = p.x.copy(); y_best_global = y
    history = [[p.x.copy() for p in population]]
    best_vals = [y_best_global]
    for _ in range(k_max):
        for p in population:
            n = len(p.x)
            r1, r2 = np.random.rand(n), np.random.rand(n)
            p.x = p.x + p.v
            p.v = w*p.v + c1*r1*(p.x_best - p.x) + c2*r2*(x_best_global - p.x)
            y = f(p.x)
            if y < y_best_global:
                x_best_global = p.x.copy(); y_best_global = y
            if y < f(p.x_best):
                p.x_best = p.x.copy()
        history.append([p.x.copy() for p in population])
        best_vals.append(y_best_global)
    return history, best_vals


def firefly_history(f, population, k_max, beta=1.0, alpha=0.2,
                    brightness=lambda r: np.exp(-r**2)):
    """Returns (history_of_populations, best_per_iter)."""
    population = [np.array(x, dtype=float) for x in population]
    d = len(population[0])
    history = [[x.copy() for x in population]]
    best_vals = [min(f(x) for x in population)]
    for _ in range(k_max):
        for i in range(len(population)):
            for j in range(len(population)):
                if f(population[j]) < f(population[i]):
                    r = np.linalg.norm(population[j] - population[i])
                    noise = np.random.randn(d)
                    population[i] = (population[i]
                                     + beta * brightness(r) * (population[j] - population[i])
                                     + alpha * noise)
        history.append([x.copy() for x in population])
        best_vals.append(min(f(x) for x in population))
    return history, best_vals


def cuckoo_history(f, population, k_max, p_a=0.25, cauchy_scale=0.5):
    """Returns (history_of_positions, best_per_iter)."""
    import copy
    population = copy.deepcopy(population)
    m = len(population); n = len(population[0].x)
    a = max(1, round(m * p_a))
    history = [[nest.x.copy() for nest in population]]
    best_vals = [min(nest.y for nest in population)]
    for _ in range(k_max):
        i = np.random.randint(0, m); j = np.random.randint(0, m)
        step = np.array([cauchy_step(cauchy_scale) for _ in range(n)])
        x_new = population[j].x + step; y_new = f(x_new)
        if y_new < population[i].y:
            population[i] = Nest(x=x_new, y=y_new)
        population.sort(key=lambda nest: nest.y)
        for bad_idx in range(m - a, m):
            src_idx = np.random.randint(0, m - a)
            step = np.array([cauchy_step(cauchy_scale) for _ in range(n)])
            x_new = population[src_idx].x + step
            population[bad_idx] = Nest(x=x_new, y=f(x_new))
        history.append([nest.x.copy() for nest in population])
        best_vals.append(population[0].y)
    return history, best_vals


# ── PLOT FUNCTIONS ─────────────────────────────────────────────────────────────

def plot_fig1_initialization():
    """Figure 1 — three sampling strategies side by side."""
    np.random.seed(7)
    fig = make_fig(1, 3, (14, 5), "Fig 9.1 — Population Initialization Strategies")
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35,
                            left=0.06, right=0.97, top=0.88, bottom=0.12)

    configs = [
        ("Uniform  U([-2,2]²)", "darkcyan",
         rand_population_uniform(1000, np.array([-2.,-2.]), np.array([2.,2.]))),
        ("Normal  N(0, I)",     GOLD,
         rand_population_normal(1000, np.zeros(2), np.eye(2))),
        ("Cauchy  C(0, 1)",     GREEN,
         rand_population_cauchy(1000, np.zeros(2), np.ones(2))),
    ]
    for col, (title, color, pop) in enumerate(configs):
        ax = fig.add_subplot(gs[col])
        pts = np.array(pop)
        # clip for visibility
        pts = pts[np.abs(pts[:,0]) < 8]; pts = pts[np.abs(pts[:,1]) < 8]
        ax.scatter(pts[:,0], pts[:,1], s=4, alpha=0.4, color=color, linewidths=0)
        apply_dark_style(ax, title, "x₁", "x₂" if col == 0 else "")
        ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
        # draw bounding box for uniform
        if col == 0:
            for xs, ys in [([-2,2,-2,2,-2],[-2,-2,2,2,-2])]:
                ax.plot(xs, ys, "--", color=WHITE, lw=0.8, alpha=0.5)
    plt.tight_layout()
    _show_plot(plt.gcf())


def plot_fig2_selection():
    """Figure 2 — three selection methods bar chart."""
    np.random.seed(0)
    y = np.array([3.5, 1.2, 4.8, 0.7, 2.1])
    labels = [f"ind {i}" for i in range(len(y))]

    fig = make_fig(1, 3, (14, 4.5), "Fig 9.2 — Selection Methods")
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.4,
                            left=0.06, right=0.97, top=0.85, bottom=0.15)

    colors_bar = [TEAL, GREEN, GOLD, ORANGE, GOLD]

    # Truncation (top-2 highlighted)
    ax = fig.add_subplot(gs[0])
    sorted_idx = np.argsort(y)
    bar_colors = [GREEN if i in sorted_idx[:2] else "#444466" for i in range(len(y))]
    ax.bar(labels, y, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax.axhline(np.sort(y)[1], color='gray', lw=0.8, ls="--", alpha=0.6)
    apply_dark_style(ax, "Truncation  (k=2)", "Individual", "f(x)")

    # Tournament
    ax = fig.add_subplot(gs[1])
    k = 2
    wins = np.zeros(len(y), dtype=int)
    for _ in range(2000):
        cands = np.random.choice(len(y), k, replace=False)
        wins[cands[np.argmin(y[cands])]] += 1
    norm_wins = wins / wins.max()
    bar_colors2 = [plt.cm.YlOrRd(v) for v in norm_wins]
    ax.bar(labels, wins, color=bar_colors2, edgecolor="white", linewidth=0.5)
    apply_dark_style(ax, "Tournament  (k=2)", "Individual", "Selection count")

    # Roulette
    ax = fig.add_subplot(gs[2])
    fitness = np.max(y) - y
    probs = fitness / fitness.sum()
    ax.bar(labels, probs, color=[plt.cm.plasma(p/probs.max()) for p in probs],
           edgecolor="white", linewidth=0.5)
    apply_dark_style(ax, "Roulette Wheel", "Individual", "Selection probability")

    plt.tight_layout()
    _show_plot(plt.gcf())


def plot_fig3_crossover():
    """Figure 3 — crossover schemes as heatmaps of binary strings."""
    np.random.seed(3)
    a = np.array([1,0,1,0,1,0,1,0], dtype=float)
    b = np.array([0,1,0,1,0,1,0,1], dtype=float)

    sp  = crossover_single_point(a.astype(bool), b.astype(bool)).astype(float)
    tp  = crossover_two_point(a.astype(bool), b.astype(bool)).astype(float)
    uni = crossover_uniform(a.astype(bool), b.astype(bool)).astype(float)

    fig = make_fig(1, 1, (12, 4), "Fig 9.3 — Crossover Methods")
    ax  = fig.add_subplot(111)
    ax.set_facecolor(DARK)

    rows = np.array([a, b, sp, tp, uni])
    row_labels = ["Parent A", "Parent B",
                  "Single-Point Child", "Two-Point Child", "Uniform Child"]

    cmap = LinearSegmentedColormap.from_list("bw", [MID, TEAL])
    ax.imshow(rows, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    # gene index labels
    ax.set_xticks(range(8)); ax.set_xticklabels([f"g{i}" for i in range(8)], color=WHITE, fontsize=9)
    ax.set_yticks(range(5)); ax.set_yticklabels(row_labels, color=WHITE, fontsize=9)
    ax.tick_params(colors=WHITE)
    for spine in ax.spines.values(): spine.set_edgecolor(ACCENT)

    # value annotations
    for r in range(5):
        for c in range(8):
            ax.text(c, r, str(int(rows[r,c])), ha="center", va="center",
                    color=WHITE, fontsize=10, fontweight="bold")

    ax.set_title("Crossover Methods — gene=1 shown in teal", color=WHITE,
                 fontsize=11, fontweight="bold", pad=8)

    plt.tight_layout()
    _show_plot(plt.gcf())


def plot_fig4_ga():
    """Figure 4 — GA convergence: 4 snapshot generations on Sphere."""
    np.random.seed(42)
    m_ga, k_max_ga = 80, 20
    a_ga = np.array([-3.,-3.]); b_ga = np.array([3.,3.])
    pop_ga = rand_population_uniform(m_ga, a_ga, b_ga)

    history = genetic_algorithm_history(
        f            = sphere,
        population   = pop_ga,
        k_max        = k_max_ga,
        select_fn    = lambda y: select_truncation(y, k=10),
        crossover_fn = crossover_single_point,
        mutate_fn    = lambda c: mutate_gaussian(np.array(c, dtype=float), 0.5)
    )

    snaps = [0, 5, 10, 20]
    fig   = make_fig(1, 4, (16, 5), "Fig 9.4 — Genetic Algorithm on Sphere  f(x)=‖x‖²")
    gs    = gridspec.GridSpec(1, 4, figure=fig, wspace=0.25,
                              left=0.05, right=0.97, top=0.88, bottom=0.12)
    for col, gen in enumerate(snaps):
        ax  = fig.add_subplot(gs[col])
        pop = np.array(history[gen])
        contour_bg(ax, sphere, (-3.5, 3.5), (-3.5, 3.5), cmap="magma")
        vals = np.array([sphere(x) for x in history[gen]])
        sc   = ax.scatter(pop[:,0], pop[:,1], c=vals, cmap="cool",
                          s=30, edgecolors="white", linewidths=0.4, vmin=0, vmax=18)
        ax.scatter(0, 0, marker="*", s=200, color=GOLD, zorder=5, label="Optimum")
        apply_dark_style(ax, f"Generation {gen}", "x₁", "x₂" if col == 0 else "")
        ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
    fig.colorbar(sc, ax=fig.axes, shrink=0.7, label="f(x)", pad=0.01).ax.yaxis.label.set_color("black")

    plt.tight_layout()
    _show_plot(plt.gcf())


def plot_fig5_de():
    """Figure 5 — DE snapshots + convergence curve on Ackley."""
    np.random.seed(5)
    pop_de = rand_population_uniform(30, np.array([-5.,-5.]), np.array([5.,5.]))
    history, best_vals = differential_evolution_history(
        ackley, pop_de, k_max=100, p=0.5, w=0.8)

    snaps = [0, 10, 30, 100]
    fig   = make_fig(2, 4, (16, 8),
                     "Fig 9.5 — Differential Evolution on Ackley")
    gs    = gridspec.GridSpec(2, 4, figure=fig, wspace=0.28, hspace=0.4,
                              left=0.06, right=0.97, top=0.90, bottom=0.08)

    for col, it in enumerate(snaps):
        ax  = fig.add_subplot(gs[0, col])
        pop = np.array(history[it])
        contour_bg(ax, ackley, (-5,5), (-5,5), cmap="viridis")
        vals = np.array([ackley(x) for x in history[it]])
        sc   = ax.scatter(pop[:,0], pop[:,1], c=vals, cmap="hot",
                          s=35, edgecolors="white", linewidths=0.3,
                          vmin=0, vmax=14)
        ax.scatter(0, 0, marker="*", s=200, color=GOLD, zorder=5)
        apply_dark_style(ax, f"Iter {it}", "x₁", "x₂" if col == 0 else "")
        ax.set_xlim(-5,5); ax.set_ylim(-5,5)

    # Convergence curve
    ax_conv = fig.add_subplot(gs[1, :])
    ax_conv.plot(best_vals, color=TEAL, lw=2)
    ax_conv.fill_between(range(len(best_vals)), best_vals, alpha=0.15, color=TEAL)
    ax_conv.set_yscale("log")
    apply_dark_style(ax_conv, "Convergence — best f(x) per iteration",
                     "Iteration", "f(x)  [log scale]")

    plt.tight_layout()
    _show_plot(plt.gcf())


def plot_fig6_pso():
    """Figure 6 — PSO snapshots + convergence on Ackley."""
    np.random.seed(6)
    def make_particle(x):
        return Particle(x=x.copy(), v=np.zeros_like(x), x_best=x.copy())
    pop_pso = [make_particle(x)
               for x in rand_population_uniform(30, np.array([-5.,-5.]),
                                                np.array([5.,5.]))]
    history, best_vals = pso_history(ackley, pop_pso, k_max=100, w=0.5, c1=1.5, c2=1.5)

    snaps = [0, 10, 30, 100]
    fig   = make_fig(2, 4, (16, 8),
                     "Fig 9.6 — Particle Swarm Optimization on Ackley")
    gs    = gridspec.GridSpec(2, 4, figure=fig, wspace=0.28, hspace=0.4,
                              left=0.06, right=0.97, top=0.90, bottom=0.08)

    for col, it in enumerate(snaps):
        ax  = fig.add_subplot(gs[0, col])
        pop = np.array(history[it])
        contour_bg(ax, ackley, (-5,5), (-5,5), cmap="viridis")
        ax.scatter(pop[:,0], pop[:,1], s=40, color=TEAL,
                   edgecolors="white", linewidths=0.4, zorder=3)
        ax.scatter(0, 0, marker="*", s=200, color=GOLD, zorder=5)
        apply_dark_style(ax, f"Iter {it}", "x₁", "x₂" if col == 0 else "")
        ax.set_xlim(-5,5); ax.set_ylim(-5,5)

    ax_conv = fig.add_subplot(gs[1, :])
    ax_conv.plot(best_vals, color=ORANGE, lw=2)
    ax_conv.fill_between(range(len(best_vals)), best_vals, alpha=0.15, color=ORANGE)
    ax_conv.set_yscale("log")
    apply_dark_style(ax_conv, "Convergence — global best f(x) per iteration",
                     "Iteration", "f(x)  [log scale]")

    plt.tight_layout()
    _show_plot(plt.gcf())


def plot_fig7_firefly():
    """Figure 7 — Firefly snapshots + convergence on Sphere."""
    np.random.seed(8)
    pop_ff = rand_population_uniform(20, np.array([-5.,-5.]), np.array([5.,5.]))
    history, best_vals = firefly_history(sphere, pop_ff, k_max=60,
                                         beta=1.0, alpha=0.3)

    snaps = [0, 5, 20, 60]
    fig   = make_fig(2, 4, (16, 8),
                     "Fig 9.7 — Firefly Algorithm on Sphere  f(x)=‖x‖²")
    gs    = gridspec.GridSpec(2, 4, figure=fig, wspace=0.28, hspace=0.4,
                              left=0.06, right=0.97, top=0.90, bottom=0.08)

    for col, it in enumerate(snaps):
        ax  = fig.add_subplot(gs[0, col])
        pop = np.array(history[it])
        contour_bg(ax, sphere, (-5,5), (-5,5), cmap="inferno")
        vals = np.array([sphere(x) for x in history[it]])
        ax.scatter(pop[:,0], pop[:,1], c=vals, cmap="YlOrRd_r",
                   s=60, edgecolors="white", linewidths=0.4, vmin=0, vmax=50)
        ax.scatter(0, 0, marker="*", s=200, color=GREEN, zorder=5)
        apply_dark_style(ax, f"Iter {it}", "x₁", "x₂" if col == 0 else "")
        ax.set_xlim(-5,5); ax.set_ylim(-5,5)

    ax_conv = fig.add_subplot(gs[1, :])
    ax_conv.plot(best_vals, color=GREEN, lw=2)
    ax_conv.fill_between(range(len(best_vals)), best_vals, alpha=0.15, color=GREEN)
    ax_conv.set_yscale("log")
    apply_dark_style(ax_conv, "Convergence — best f(x) per iteration",
                     "Iteration", "f(x)  [log scale]")

    plt.tight_layout()
    _show_plot(plt.gcf())


def plot_fig8_cuckoo():
    """Figure 8 — Cuckoo Search snapshots + convergence on Sphere."""
    np.random.seed(9)
    pop_cs_init = rand_population_uniform(20, np.array([-5.,-5.]), np.array([5.,5.]))
    pop_cs = [Nest(x=x, y=sphere(x)) for x in pop_cs_init]
    history, best_vals = cuckoo_history(sphere, pop_cs, k_max=200,
                                        p_a=0.25, cauchy_scale=0.5)

    snaps = [0, 20, 80, 200]
    fig   = make_fig(2, 4, (16, 8),
                     "Fig 9.8 — Cuckoo Search on Sphere  f(x)=‖x‖²")
    gs    = gridspec.GridSpec(2, 4, figure=fig, wspace=0.28, hspace=0.4,
                              left=0.06, right=0.97, top=0.90, bottom=0.08)

    for col, it in enumerate(snaps):
        ax  = fig.add_subplot(gs[0, col])
        pop = np.array(history[it])
        # clip extreme Cauchy steps for visibility
        pop_vis = np.clip(pop, -8, 8)
        contour_bg(ax, sphere, (-5,5), (-5,5), cmap="cividis")
        vals = np.array([sphere(p) for p in history[it]])
        ax.scatter(pop_vis[:,0], pop_vis[:,1], c=np.clip(vals,0,50),
                   cmap="autumn_r", s=50, edgecolors=WHITE,
                   linewidths=0.4, vmin=0, vmax=50)
        ax.scatter(0, 0, marker="*", s=200, color=GOLD, zorder=5)
        apply_dark_style(ax, f"Iter {it}", "x₁", "x₂" if col == 0 else "")
        ax.set_xlim(-6,6); ax.set_ylim(-6,6)

    ax_conv = fig.add_subplot(gs[1, :])
    ax_conv.plot(best_vals, color=GOLD, lw=2)
    ax_conv.fill_between(range(len(best_vals)), best_vals, alpha=0.15, color=GOLD)
    ax_conv.set_yscale("log")
    apply_dark_style(ax_conv, "Convergence — best f(x) per iteration",
                     "Iteration", "f(x)  [log scale]")

    plt.tight_layout()
    _show_plot(plt.gcf())


def plot_fig9_hybrid():
    """Figure 9 — Lamarckian vs Baldwinian on two-hump function."""
    np.random.seed(11)
    xs = np.linspace(-1.5, 5, 500)
    ys = np.array([-np.exp(-x**2) - 2*np.exp(-(x-3)**2) for x in xs])
    pop_hybrid = [np.array([np.random.uniform(-0.5, 0.5)]) for _ in range(8)]

    pop_lam  = lamarckian_step(two_hump, pop_hybrid.copy(), step_size=0.03)
    pop_bal, _ = baldwinian_step(two_hump, pop_hybrid.copy(), step_size=0.03)

    fig = make_fig(1, 2, (13, 5), "Fig 9.9 — Lamarckian vs Baldwinian Learning")
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35,
                            left=0.07, right=0.97, top=0.88, bottom=0.14)

    for col, (title, pop, color, note) in enumerate([
        ("Lamarckian\n(positions REPLACED)", pop_lam,  GOLD,
         "Stuck near local min"),
        ("Baldwinian\n(positions UNCHANGED)", pop_bal, TEAL,
         "Can still escape"),
    ]):
        ax = fig.add_subplot(gs[col])
        ax.plot(xs, ys, color='black', lw=1.5, alpha=0.8)
        ax.fill_between(xs, ys, ys.min()-0.1, alpha=0.1, color='gray')
        ax.axvline(0,  color="#999999", lw=1.0, ls="--", label="local min")
        ax.axvline(3,  color=GREEN,    lw=0.8, ls="--", label="global min")
        pts_x = [p[0] for p in pop]
        pts_y = [two_hump(p) for p in pop]
        ax.scatter(pts_x, pts_y, s=80, color=color, zorder=5,
                   edgecolors="gray", linewidths=0.6)
        ax.scatter([p[0] for p in pop_hybrid],
                   [two_hump(p) for p in pop_hybrid],
                   s=40, color="#888888", zorder=4, marker="x", linewidths=1.5,
                   label="initial pos")
        # arrows from initial to final
        for init, final in zip(pop_hybrid, pop):
            ax.annotate("", xy=(final[0], two_hump(final)),
                        xytext=(init[0], two_hump(init)),
                        arrowprops=dict(arrowstyle="->", color=color,
                                        lw=1.2, connectionstyle="arc3,rad=0.2"))
        ax.legend(fontsize=7, facecolor="white", labelcolor="black", framealpha=0.7)
        apply_dark_style(ax, title, "x", "f(x)")
        ax.set_xlim(-1.5, 5); ax.set_ylim(ys.min()-0.15, 0.4)
        ax.text(0.5, 0.07, note, transform=ax.transAxes, color=color,
                fontsize=8, ha="center", style="italic")

    plt.tight_layout()
    _show_plot(plt.gcf())


def plot_fig10_summary(results: dict):
    """Figure 10 — Final comparison bar chart of all algorithms."""
    fig = make_fig(1, 1, (11, 5), "Fig 9.10 — Algorithm Comparison  (lower = better)")
    ax  = fig.add_subplot(111)
    ax.set_facecolor(MID)

    names  = list(results.keys())
    values = [max(v, 1e-9) for v in results.values()]   # avoid log(0)
    colors = [TEAL, GOLD, ORANGE, GREEN, "#c77dff"]

    bars = ax.bar(names, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_yscale("log")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() * 1.4,
                f"{val:.2e}", ha="center", va="bottom",
                color='black', fontsize=8)
    apply_dark_style(ax, "", "Algorithm", "Best f(x)  [log scale]")
    ax.axhline(1e-6, color='gray', lw=0.7, ls="--", alpha=0.4, label="1e-6 threshold")
    ax.tick_params(axis="x", labelsize=9)
    for spine in ax.spines.values(): spine.set_edgecolor(ACCENT)
    ax.legend(facecolor="white", labelcolor="black", fontsize=8, framealpha=0.7)

    plt.tight_layout()
    _show_plot(plt.gcf())


# ── MAIN ───────────────────────────────────────────────────────────────────────

def run_all():
    np.random.seed(42); random.seed(42)

    separator("9.1 INITIALIZATION — Population Sampling")
    a_bounds = np.array([-2.,-2.]); b_bounds = np.array([2.,2.])
    pop_uniform = rand_population_uniform(5, a_bounds, b_bounds)
    print("\n[Uniform] 5 design points sampled from [-2,2]²:")
    for pt in pop_uniform: print(f"  {pt.round(4)}")
    pop_normal = rand_population_normal(5, np.zeros(2), np.eye(2))
    print("\n[Normal]  5 design points sampled from N(0, I):")
    for pt in pop_normal: print(f"  {pt.round(4)}")
    pop_cauchy = rand_population_cauchy(5, np.zeros(2), np.ones(2))
    print("\n[Cauchy]  5 design points sampled with Cauchy(0,1):")
    for pt in pop_cauchy: print(f"  {pt.round(4)}")
    pop_binary = rand_population_binary(3, n=8)
    print("\n[Binary]  3 binary chromosomes of length 8:")
    for ch in pop_binary: print(f"  {ch.astype(int)}")

    separator("9.2.3 SELECTION METHODS")
    y_demo = np.array([3.5, 1.2, 4.8, 0.7, 2.1])
    print(f"\nObjective values: {y_demo}")
    print(f"[Truncation k=2] pairs: {select_truncation(y_demo, k=2)[:3]}...")
    print(f"[Tournament k=2] pairs: {select_tournament(y_demo, k=2)[:3]}...")
    print(f"[Roulette      ] pairs: {select_roulette(y_demo)[:3]}...")

    separator("9.2.4 CROSSOVER METHODS")
    a_chr = np.array([1,0,1,0,1,0,1,0], dtype=bool)
    b_chr = np.array([0,1,0,1,0,1,0,1], dtype=bool)
    print(f"\nParent A: {a_chr.astype(int)}")
    print(f"Parent B: {b_chr.astype(int)}")
    print(f"[Single-Point ] Child: {crossover_single_point(a_chr, b_chr).astype(int)}")
    print(f"[Two-Point    ] Child: {crossover_two_point(a_chr, b_chr).astype(int)}")
    print(f"[Uniform      ] Child: {crossover_uniform(a_chr, b_chr).astype(int)}")
    a_r = np.array([1.,2.,3.]); b_r = np.array([4.,5.,6.])
    print(f"[Interpolation] Child: {crossover_interpolation(a_r, b_r, 0.5)}")

    separator("9.2.5 MUTATION METHODS")
    child_bin  = np.array([1,0,1,1,0,0,1,0], dtype=bool)
    child_real = np.array([1.5, -0.3, 2.7])
    print(f"\n[Bitwise  λ=0.2] Before: {child_bin.astype(int)}")
    print(f"                 After:  {mutate_bitwise(child_bin, 0.2).astype(int)}")
    print(f"\n[Gaussian σ=0.1] Before: {child_real}")
    print(f"                 After:  {mutate_gaussian(child_real, 0.1).round(4)}")

    separator("9.2 GENETIC ALGORITHM — Minimize ||x||")
    m_ga, k_max_ga = 100, 20
    pop_ga = rand_population_uniform(m_ga, np.array([-3.,-3.]), np.array([3.,3.]))
    result_ga = genetic_algorithm(
        f=lambda x: np.linalg.norm(x), population=pop_ga, k_max=k_max_ga,
        select_fn=lambda y: select_truncation(y, k=10),
        crossover_fn=crossover_single_point,
        mutate_fn=lambda c: mutate_gaussian(np.array(c, dtype=float), 0.5))
    print(f"\nBest x: {result_ga.round(6)}   f(x): {np.linalg.norm(result_ga):.6f}")

    separator("9.3 DIFFERENTIAL EVOLUTION — Ackley")
    pop_de = rand_population_uniform(30, np.array([-5.,-5.]), np.array([5.,5.]))
    best_de = differential_evolution(ackley, pop_de, k_max=200, p=0.5, w=0.8)
    print(f"\nBest x: {best_de.round(6)}   Ackley(x): {ackley(best_de):.6f}")

    separator("9.4 PARTICLE SWARM OPTIMIZATION — Ackley")
    def make_particle(x): return Particle(x=x.copy(), v=np.zeros_like(x), x_best=x.copy())
    pop_pso = [make_particle(x) for x in
               rand_population_uniform(30, np.array([-5.,-5.]), np.array([5.,5.]))]
    pop_pso_res = particle_swarm_optimization(ackley, pop_pso, k_max=200, w=0.5, c1=1.5, c2=1.5)
    best_pso = min(pop_pso_res, key=lambda p: ackley(p.x_best))
    print(f"\nBest x: {best_pso.x_best.round(6)}   Ackley(x): {ackley(best_pso.x_best):.6f}")

    separator("9.5 FIREFLY ALGORITHM — Sphere")
    pop_ff = rand_population_uniform(20, np.array([-5.,-5.]), np.array([5.,5.]))
    best_ff = firefly(sphere, pop_ff, k_max=50, beta=1.0, alpha=0.2)
    print(f"\nBest x: {best_ff.round(6)}   Sphere(x): {sphere(best_ff):.6f}")

    separator("9.6 CUCKOO SEARCH — Sphere")
    pop_cs = [Nest(x=x, y=sphere(x)) for x in
              rand_population_uniform(20, np.array([-5.,-5.]), np.array([5.,5.]))]
    pop_cs_res = cuckoo_search(sphere, pop_cs, k_max=200, p_a=0.25, cauchy_scale=0.5)
    best_cs = pop_cs_res[0]
    print(f"\nBest x: {best_cs.x.round(6)}   Sphere(x): {best_cs.y:.6f}")

    separator("9.7 HYBRID METHODS — Lamarckian vs Baldwinian")
    pop_hybrid = [np.array([np.random.uniform(-0.5,0.5)]) for _ in range(6)]
    pop_lam  = lamarckian_step(two_hump, pop_hybrid.copy(), step_size=0.05)
    pop_bal, perceived = baldwinian_step(two_hump, pop_hybrid.copy(), step_size=0.05)
    print(f"\n[Lamarckian] positions: {[round(x[0],3) for x in pop_lam]}")
    print(f"[Baldwinian] positions: {[round(x[0],3) for x in pop_bal]}  (unchanged)")
    print(f"             perceived: {[round(v,4) for v in perceived]}")

    separator("GENERATING ALL PLOTS")
    print("\nRendering figures — please wait...")
    plot_fig1_initialization()
    plot_fig2_selection()
    plot_fig3_crossover()
    plot_fig4_ga()
    plot_fig5_de()
    plot_fig6_pso()
    plot_fig7_firefly()
    plot_fig8_cuckoo()
    plot_fig9_hybrid()

    results = {
        "Genetic\nAlgorithm"  : float(np.linalg.norm(result_ga)),
        "Differential\nEvolution" : float(ackley(best_de)),
        "Particle\nSwarm"     : float(ackley(best_pso.x_best)),
        "Firefly\nAlgorithm"  : float(sphere(best_ff)),
        "Cuckoo\nSearch"      : float(best_cs.y),
    }
    plot_fig10_summary(results)

    vals = list(results.values())
    separator("SUMMARY")
    print(f"""
  Algorithm                | Problem       | Best f(x)
  -------------------------|---------------|----------
  Genetic Algorithm        | Sphere (2D)   | {vals[0]:.6f}
  Differential Evolution   | Ackley (2D)   | {vals[1]:.6f}
  Particle Swarm Optim.    | Ackley (2D)   | {vals[2]:.6f}
  Firefly Algorithm        | Sphere (2D)   | {vals[3]:.6f}
  Cuckoo Search            | Sphere (2D)   | {vals[4]:.6f}

""")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_all()

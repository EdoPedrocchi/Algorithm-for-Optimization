# Algorithm-for-Optimization
# Chapter 9: Population Methods

Previous chapters have focused on methods where a single design point is moved incrementally toward a minimum. This chapter presents a variety of population methods that involve optimization using a collection of design points, called **individuals**. Having a large number of individuals distributed throughout the design space can help the algorithm avoid becoming stuck in a local minimum. Information at different points in the design space can be shared between individuals to globally optimize the objective function. Most population methods are stochastic in nature, and it is generally easy to parallelize the computation.

---

## 9.1 Initialization

Population methods begin with an initial population, just as descent methods require an initial design point. The initial population should be spread over the design space to increase the chances that the samples are close to the best regions.

We can often constrain the design variables to a region of interest consisting of a hyperrectangle defined by lower and upper bounds **a** and **b**. Initial populations can be sampled from a uniform distribution for each coordinate:

$$x_i^{(j)} \sim U(a_i, b_i)$$

where $x^{(j)}$ is the $j$th individual in the population.

### Algorithm 9.1 — Uniform Population Sampling

A method for sampling an initial population of `m` design points over a uniform hyperrectangle with lower-bound vector `a` and upper-bound vector `b`.

```julia
function rand_population_uniform(m, a, b)
    d = length(a)
    return [a+rand(d).*(b-a) for i in 1:m]
end
```

### Algorithm 9.2 — Normal Population Sampling

A method for sampling an initial population of `m` design points using a multivariate normal distribution with mean `μ` and covariance `Σ`.

```julia
using Distributions
function rand_population_normal(m, μ, Σ)
    D = MvNormal(μ,Σ)
    return [rand(D) for i in 1:m]
end
```

Uniform and normal distributions limit the covered design space to a concentrated region. The **Cauchy distribution** has an unbounded variance and can cover a much broader space.

### Algorithm 9.3 — Cauchy Population Sampling

A method for sampling an initial population of `m` design points using a Cauchy distribution with location `μ` and scale `σ` for each dimension.

```julia
using Distributions
function rand_population_cauchy(m, μ, σ)
    n = length(μ)
    return [[rand(Cauchy(μ[j],σ[j])) for j in 1:n] for i in 1:m]
end
```

---

## 9.2 Genetic Algorithms

**Genetic algorithms** borrow inspiration from biological evolution, where fitter individuals are more likely to pass on their genes to the next generation. An individual's fitness for reproduction is inversely related to the value of the objective function at that point. The design point associated with an individual is represented as a **chromosome**. At each generation, the chromosomes of the fitter individuals are passed on to the next generation after undergoing the genetic operations of **crossover** and **mutation**.

### Algorithm 9.4 — Genetic Algorithm

The genetic algorithm, which takes an objective function `f`, an initial `population`, number of iterations `k_max`, a `SelectionMethod` S, a `CrossoverMethod` C, and a `MutationMethod` M.

```julia
function genetic_algorithm(f, population, k_max, S, C, M)
    for k in 1 : k_max
        parents = select(S, f.(population))
        children = [crossover(C,population[p[1]],population[p[2]])
                    for p in parents]
        population .= mutate.(Ref(M), children)
    end
    population[argmin(f.(population))]
end
```

### 9.2.1 Chromosomes

There are several ways to represent chromosomes. The simplest is the **binary string chromosome**, a representation similar to the way DNA is encoded. A random binary string of length `d` can be generated using `bitrand(d)`.

Binary strings are often used due to the ease of expressing crossover and mutation. It is often more natural to represent a chromosome using a list of real values. Such **real-valued chromosomes** are vectors in $\mathbb{R}^d$ that directly correspond to points in the design space.

### 9.2.2 Initialization

Genetic algorithms start with a random initial population. Binary string chromosomes are typically initialized using random bit strings.

### Algorithm 9.5 — Binary Population Sampling

A method for sampling random starting populations of `m` bit-string chromosomes of length `n`.

```julia
rand_population_binary(m, n) = [bitrand(n) for i in 1:m]
```

### 9.2.3 Selection

Selection is the process of choosing chromosomes to use as parents for the next generation. For a population with `m` chromosomes, a selection method will produce a list of `m` parental pairs for the `m` children of the next generation.

There are several approaches for biasing the selection toward the fittest:

- **Truncation selection** — sample parents from among the best `k` chromosomes in the population.
- **Tournament selection** — each parent is the fittest out of `k` randomly chosen chromosomes.
- **Roulette wheel selection** (fitness proportionate selection) — each parent is chosen with a probability proportional to its performance relative to the population. The fitness of individual $i$ is assigned according to $\max\{y^{(1)}, \ldots, y^{(m)}\} - y^{(i)}$.

### Algorithm 9.6 — Selection Methods

```julia
abstract type SelectionMethod end

# Pick pairs randomly from top k parents
struct TruncationSelection <: SelectionMethod
    k # top k to keep
end
function select(t::TruncationSelection, y)
    p = sortperm(y)
    return [p[rand(1:t.k, 2)] for i in y]
end

# Pick parents by choosing best among random subsets
struct TournamentSelection <: SelectionMethod
    k
end
function select(t::TournamentSelection, y)
    getparent() = begin
        p = randperm(length(y))
        p[argmin(y[p[1:t.k]])]
    end
    return [[getparent(), getparent()] for i in y]
end

# Sample parents proportionately to fitness
struct RouletteWheelSelection <: SelectionMethod end
function select(::RouletteWheelSelection, y)
    y = maximum(y) .- y
    cat = Categorical(normalize(y, 1))
    return [rand(cat, 2) for i in y]
end
```

### 9.2.4 Crossover

Crossover combines the chromosomes of parents to form children. There are several crossover schemes:

- **Single-point crossover** — the first portion of parent A's chromosome forms the first portion of the child, and the latter portion of parent B's chromosome forms the latter part. The crossover point is determined uniformly at random.
- **Two-point crossover** — uses two random crossover points.
- **Uniform crossover** — each bit has a fifty percent chance of coming from either parent.

For real-valued chromosomes, values can also be linearly interpolated between the parents:

$$x \leftarrow (1 - \lambda)x_a + \lambda x_b$$

where $\lambda$ is a scalar parameter typically set to one-half.

### Algorithm 9.7 — Crossover Methods

```julia
abstract type CrossoverMethod end

struct SinglePointCrossover <: CrossoverMethod end
function crossover(::SinglePointCrossover, a, b)
    i = rand(1:length(a))
    return vcat(a[1:i], b[i+1:end])
end

struct TwoPointCrossover <: CrossoverMethod end
function crossover(::TwoPointCrossover, a, b)
    n = length(a)
    i, j = rand(1:n, 2)
    if i > j
        (i,j) = (j,i)
    end
    return vcat(a[1:i], b[i+1:j], a[j+1:n])
end

struct UniformCrossover <: CrossoverMethod end
function crossover(::UniformCrossover, a, b)
    child = copy(a)
    for i in 1 : length(a)
        if rand() < 0.5
            child[i] = b[i]
        end
    end
    return child
end
```

### Algorithm 9.8 — Interpolation Crossover

A crossover method for real-valued chromosomes which performs linear interpolation between the parents.

```julia
struct InterpolationCrossover <: CrossoverMethod
    λ
end
crossover(C::InterpolationCrossover, a, b) = (1-C.λ)*a + C.λ*b
```

### 9.2.5 Mutation

If new chromosomes were produced only through crossover, many traits not present in the initial random population could never occur, and the most-fit genes could saturate the population. **Mutation** allows new traits to spontaneously appear, allowing the genetic algorithm to explore more of the state space. Child chromosomes undergo mutation after crossover.

Each bit in a binary-valued chromosome typically has a small probability of being flipped. For a chromosome with `m` bits, this mutation rate is typically set to `1/m`, yielding an average of one mutation per child chromosome. Mutation for real-valued chromosomes is more commonly implemented by adding zero-mean Gaussian noise.

### Algorithm 9.9 — Mutation Methods

The bitwise mutation method for binary string chromosomes and the Gaussian mutation method for real-valued chromosomes. Here, `λ` is the mutation rate, and `σ` is the standard deviation.

```julia
abstract type MutationMethod end

struct BitwiseMutation <: MutationMethod
    λ
end
function mutate(M::BitwiseMutation, child)
    return [rand() < M.λ ? !v : v for v in child]
end

struct GaussianMutation <: MutationMethod
    σ
end
function mutate(M::GaussianMutation, child)
    return child + randn(length(child))*M.σ
end
```

### Example 9.1 — Genetic Algorithm Demo

Demonstration of using a genetic algorithm for optimizing a simple function.

```julia
import Random: seed!
import LinearAlgebra: norm
seed!(0) # set random seed for reproducible results
f = x->norm(x)
m = 100 # population size
k_max = 10 # number of iterations
population = rand_population_uniform(m, [-3, 3], [3,3])
S = TruncationSelection(10) # select top 10
C = SinglePointCrossover()
M = GaussianMutation(0.5) # small mutation rate
x = genetic_algorithm(f, population, k_max, S, C, M)
@show x
```

```
x = [-0.00994906141228906, -0.05198433759424115]
```

---

## 9.3 Differential Evolution

**Differential evolution** attempts to improve each individual in the population by recombining other individuals according to a simple formula. It is parameterized by a crossover probability `p` and a differential weight `w`. Typically, `w` is between 0.4 and 1. For each individual `x`:

1. Choose three random distinct individuals **a**, **b**, and **c**.
2. Construct an interim design $z = a + w \cdot (b - c)$.
3. Choose a random dimension $j \in [1, \ldots, n]$.
4. Construct the candidate individual $x'$ using binary crossover:

$$x'_i = \begin{cases} z_i & \text{if } i = j \text{ or with probability } p \\ x_i & \text{otherwise} \end{cases}$$

5. Insert the better design between $x$ and $x'$ into the next generation.

### Algorithm 9.10 — Differential Evolution

Differential evolution, which takes an objective function `f`, a `population`, a number of iterations `k_max`, a crossover probability `p`, and a differential weight `w`. The best individual is returned.

```julia
using StatsBase
function differential_evolution(f, population, k_max; p=0.5, w=1)
    n, m = length(population[1]), length(population)
    for k in 1 : k_max
        for (k,x) in enumerate(population)
            a, b, c = sample(population,
                Weights([j!=k for j in 1:m]), 3, replace=false)
            z = a + w*(b-c)
            j = rand(1:n)
            x′ = [i == j || rand() < p ? z[i] : x[i] for i in 1:n]
            if f(x′) < f(x)
                x[:] = x′
            end
        end
    end
    return population[argmin(f.(population))]
end
```

---

## 9.4 Particle Swarm Optimization

**Particle swarm optimization** introduces momentum to accelerate convergence toward minima. Each individual, or *particle*, in the population keeps track of its current position, velocity, and the best position it has seen so far. Momentum allows an individual to accumulate speed in a favorable direction, independent of local perturbations.

### Algorithm 9.11 — Particle Struct

Each particle in particle swarm optimization has a position `x` and velocity `v` in design space and keeps track of the best design point found so far, `x_best`.

```julia
mutable struct Particle
    x
    v
    x_best
end
```

At each iteration, each individual is accelerated toward both the best position it has seen and the best position found thus far by any individual. The update equations are:

$$x^{(i)} \leftarrow x^{(i)} + v^{(i)}$$

$$v^{(i)} \leftarrow wv^{(i)} + c_1 r_1 \left(x^{(i)}_\text{best} - x^{(i)}\right) + c_2 r_2 \left(x_\text{best} - x^{(i)}\right)$$

where $x_\text{best}$ is the best location found so far over all particles; $w$, $c_1$, and $c_2$ are parameters; and $r_1$ and $r_2$ are random numbers drawn from $U(0, 1)$.

> A common strategy is to allow the inertia $w$ to decay over time.

### Algorithm 9.12 — Particle Swarm Optimization

Particle swarm optimization, which takes an objective function `f`, a list of particles `population`, a number of iterations `k_max`, an inertia `w`, and momentum coefficients `c1` and `c2`.

```julia
function particle_swarm_optimization(f, population, k_max;
                                     w=1, c1=1, c2=1)
    n = length(population[1].x)
    x_best, y_best = copy(population[1].x_best), Inf
    for P in population
        y = f(P.x)
        if y < y_best; x_best[:], y_best = P.x, y; end
    end
    for k in 1 : k_max
        for P in population
            r1, r2 = rand(n), rand(n)
            P.x += P.v
            P.v = w*P.v + c1*r1.*(P.x_best - P.x) +
                          c2*r2.*(x_best - P.x)
            y = f(P.x)
            if y < y_best; x_best[:], y_best = P.x, y; end
            if y < f(P.x_best); P.x_best[:] = P.x; end
        end
    end
    return population
end
```

---

## 9.5 Firefly Algorithm

The **firefly algorithm** was inspired by the manner in which fireflies flash their lights to attract mates. In the firefly algorithm, each individual in the population is a firefly and can flash to attract other fireflies. At each iteration, all fireflies are moved toward all more attractive fireflies. A firefly **a** is moved toward a firefly **b** with greater attraction according to:

$$a \leftarrow a + \beta I(\|b - a\|)(b - a) + \alpha \epsilon$$

where $I$ is the intensity of the attraction and $\beta$ is the source intensity. A random walk component is included as well, where $\epsilon$ is drawn from a zero-mean, unit covariance multivariate Gaussian, and $\alpha$ scales the step size.

The intensity $I$ decreases as the distance $r$ between the two fireflies increases and is defined to be 1 when $r = 0$. Several models are available:

- **Inverse square law**: $I(r) = \dfrac{1}{r^2}$

- **Exponential decay** (absorbed in medium): $I(r) = e^{-\gamma r}$

- **Gaussian drop-off** (recommended — avoids singularity at $r = 0$): $I(r) = e^{-\gamma r^2}$

### Algorithm 9.13 — Firefly Algorithm

The firefly algorithm, which takes an objective function `f`, a population `flies` of design points, a number of iterations `k_max`, a source intensity `β`, a random walk step size `α`, and an intensity function `I`. The best design point is returned.

```julia
using Distributions
function firefly(f, population, k_max;
                 β=1, α=0.1, brightness=r->exp(-r^2))
    m = length(population[1])
    N = MvNormal(Matrix(1.0I, m, m))
    for k in 1 : k_max
        for a in population, b in population
            if f(b) < f(a)
                r = norm(b-a)
                a[:] += β*brightness(r)*(b-a) + α*rand(N)
            end
        end
    end
    return population[argmin([f(x) for x in population])]
end
```

---

## 9.6 Cuckoo Search

**Cuckoo search** is another nature-inspired algorithm named after the cuckoo bird, which engages in a form of brood parasitism. Cuckoos lay their eggs in the nests of other birds; the host bird may detect the invasive egg and destroy it or establish a new nest elsewhere, or may accept and raise it.

In cuckoo search, each nest represents a design point. New design points are produced using **Lévy flights** from nests — random walks with step-lengths from a heavy-tailed distribution. A new design point can replace a nest if it has a better objective function value.

The core rules are:

1. A cuckoo will lay an egg in a randomly chosen nest.
2. The best nests with the best eggs will survive to the next generation.
3. Cuckoo eggs have a chance of being discovered by the host bird, in which case the eggs are destroyed.

Cuckoo search uses a **Cauchy distribution** for random flights, which has a heavier tail than uniform or Gaussian distributions and is more representative of animal movement patterns in the wild.

### Algorithm 9.14 — Cuckoo Search

Cuckoo search, which takes an objective function `f`, an initial set of nests `population`, a number of iterations `k_max`, percent of nests to abandon `p_a`, and flight distribution `C`.

```julia
using Distributions
mutable struct Nest
    x # position
    y # value, f(x)
end

function cuckoo_search(f, population, k_max;
                       p_a=0.1, C=Cauchy(0,1))
    m, n = length(population), length(population[1].x)
    a = round(Int, m*p_a)
    for k in 1 : k_max
        i, j = rand(1:m), rand(1:m)
        x = population[j].x + [rand(C) for k in 1 : n]
        y = f(x)
        if y < population[i].y
            population[i].x[:] = x
            population[i].y = y
        end
        p = sortperm(population, by=nest->nest.y, rev=true)
        for i in 1 : a
            j = rand(1:m-a)+a
            population[p[i]] = Nest(population[p[j]].x +
                                    [rand(C) for k in 1 : n],
                                    f(population[p[i]].x)
                                   )
        end
    end
    return population
end
```

> Other nature-inspired algorithms include the artificial bee colony, the gray wolf optimizer, the bat algorithm, glowworm swarm optimization, intelligent water drops, and harmony search. There has been some criticism of the proliferation of methods that make analogies to nature without fundamentally contributing novel methods and understanding.

---

## 9.7 Hybrid Methods

Many population methods perform well in **global search**, being able to avoid local minima and finding the best regions of the design space. However, these methods do not perform as well in **local search** compared to descent methods. Several hybrid methods (also referred to as *memetic algorithms* or *genetic local search*) have been developed to extend population methods with descent-based features.

There are two general approaches:

- **Lamarckian learning** — the population method is extended with a local search method that locally improves each individual. The original individual and its objective function value are **replaced** by the individual's optimized counterpart.

- **Baldwinian learning** — the same local search method is applied to each individual, but the results are used only to update the individual's **perceived** objective function value. Individuals are not replaced but are merely associated with optimized objective function values. Baldwinian learning can help prevent premature convergence.

### Example 9.2 — Lamarckian vs. Baldwinian Learning

Consider optimizing $f(x) = -e^{-x^2} - 2e^{-(x-3)^2}$ using a population of individuals initialized near $x = 0$.

A **Lamarckian** local search update applied to this population would move the individuals toward the local minimum, reducing the chance that future individuals escape and find the global optimum near $x = 3$.

A **Baldwinian** approach will compute the same update but leaves the original designs unchanged. The selection step will value each design according to its value from a local search, preserving diversity and improving the chances of finding the global optimum.

---

## 9.8 Summary

- Population methods use a collection of individuals in the design space to guide progression toward an optimum.
- Genetic algorithms leverage **selection**, **crossover**, and **mutations** to produce better subsequent generations.
- **Differential evolution**, **particle swarm optimization**, the **firefly algorithm**, and **cuckoo search** include rules and mechanisms for attracting design points to the best individuals in the population while maintaining suitable state space exploration.
- Population methods can be extended with local search approaches to improve convergence.

---

## 9.9 Exercises

**Exercise 9.1.** What is the motivation behind the selection operation in genetic algorithms?

**Exercise 9.2.** Why does mutation play such a fundamental role in genetic algorithms? How would we choose the mutation rate if we suspect there is a better optimal solution?

**Exercise 9.3.** If we observe that particle swarm optimization results in fast convergence to a nonglobal minimum, how might we change the parameters of the algorithm?

---

> © 2019 Massachusetts Institute of Technology, shared under a Creative Commons CC-BY-NC-ND license.  
> *Kochenderfer, M. J. & Wheeler, T. A. (2019). Algorithms for Optimization. The MIT Press.*

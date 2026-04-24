---
layout: default
---

This repository documents the theory of Population Methods for optimization, drawn primarily from Chapter 9 of Kochenderfer and Wheeler (2019), *Algorithms for Optimization*, MIT Press, supplemented with additional material. The theoretical exposition is followed by practical implementations.

## 1. Population Methods

Classical optimization methods, such as gradient descent or Newton's method, maintain and iteratively update a single design point. Their convergence depends heavily on the shape of the objective landscape near the starting point: a poorly chosen initialization can trap them in a local minimum with no mechanism to escape.

Population methods address this limitation by maintaining a collection of design points, called **individuals**, distributed across the search space. These individuals exchange information with one another, allowing the algorithm to simultaneously explore multiple regions of the domain. The result is a class of methods that is more robust to non-convexity, does not require gradient information, and is naturally suited to parallelization.

The tradeoff is computational: because fitness must be evaluated for every individual at every iteration, population methods tend to require significantly more objective function evaluations than gradient-based approaches. In practice, this cost is acceptable whenever the objective landscape is highly multimodal, when derivatives are unavailable or expensive, or when the problem dimension is moderate and the function evaluations are cheap.

---

## 1.1 Initialization

Before the first iteration, a population of $m$ individuals must be placed somewhere in the design space. The quality of this initialization influences how quickly the algorithm finds good regions: a population concentrated in one corner of the space may spend many iterations simply discovering where the better regions are.

A natural first choice is to sample uniformly from a hyperrectangle defined by lower bounds $\mathbf{a}$ and upper bounds $\mathbf{b}$:

$$x_i^{(j)} \sim U(a_i,\, b_i), \quad i = 1, \ldots, d, \quad j = 1, \ldots, m$$

This gives each region of the feasible box an equal probability of being represented in the initial population.

When prior knowledge suggests the optimum lies near some reference point $\mu$, one can instead sample from a multivariate normal distribution:

$$\mathbf{x}^{(j)} \sim \mathcal{N}(\mu,\, \Sigma)$$

The covariance matrix $\Sigma$ controls the spread. A diagonal $\Sigma$ with large diagonal entries produces a wide, diffuse cloud; a small $\Sigma$ concentrates the population near $\mu$.

A more exploratory alternative is the Cauchy distribution. Unlike the normal distribution, the Cauchy distribution has no finite variance, and its tails decay as $1/r^2$ rather than exponentially. Sampling from it tends to produce a population that is dense near the center but occasionally generates individuals that are very far from it. This heavy-tailed behavior can be beneficial when the global optimum is far from any reasonable prior guess.

<img width="1080" height="594" alt="Initialization comparison" src="https://github.com/user-attachments/assets/445d7337-4bf6-46d9-a241-5cdb46c51699" />

The figure above compares the three sampling strategies in two dimensions. The uniform sampler fills the bounding box evenly. The normal sampler concentrates mass near the origin. The Cauchy sampler has the same center but spreads occasional individuals much further out.

<img width="1236" height="583" alt="Normal vs Cauchy tails" src="https://github.com/user-attachments/assets/50a6c216-561a-4efe-a6bc-22cafcc5adc7" />

The figure above compares the tail behavior of the normal and Cauchy distributions for the same scale parameter. The heavier Cauchy tail is evident in the extreme quantiles, which is precisely what makes it useful for broad initial exploration.

---

## 1.2 Genetic Algorithms

Genetic algorithms model the optimization process as an analogue of Darwinian evolution. A population of candidate solutions evolves over successive generations. Individuals that perform well on the objective function are more likely to contribute genetic material to the next generation; individuals that perform poorly are gradually eliminated. Over time, beneficial traits accumulate and the population converges toward high-quality regions of the search space.

The algorithm cycles through three operations at each generation: selection, crossover, and mutation.

### 1.2.1 Chromosomes

Each individual in the population is encoded as a **chromosome**, a data structure that represents a point in the design space. The two most common representations are:

A **binary string chromosome** encodes each dimension as one or more bits. This representation is simple, mutation and crossover have clean interpretations, and it connects naturally to the theoretical analysis of genetic algorithms via schema theory. The length of the string determines the resolution of the encoding.

A **real-valued chromosome** is simply a vector $\mathbf{x} \in \mathbb{R}^d$ that corresponds directly to a point in the design space. This representation avoids discretization artifacts and is typically preferred for continuous optimization problems.

<img width="1150" height="122" alt="Binary chromosome" src="https://github.com/user-attachments/assets/7f2b356a-10f5-490e-b7bb-728623ba7974" />

### 1.2.2 Initialization

The initial population is generated randomly. For binary chromosomes, each bit is drawn independently from a Bernoulli(0.5) distribution. For real-valued chromosomes, individuals are typically drawn from a uniform distribution over the feasible region, as described in Section 1.1.

### 1.2.3 Selection

Selection assigns reproductive opportunities to individuals in proportion to their quality. The goal is to ensure that better individuals are more likely to become parents while still maintaining diversity in the population. For a population of $m$ individuals, the selection step produces $m$ pairs of parents, one pair per child in the next generation.

**Truncation selection** ranks all individuals by objective function value and restricts the parent pool to the top $k$. Parents are then drawn uniformly at random from this elite subset. The method is simple and computationally cheap but applies very strong selection pressure, which can cause the population to lose diversity quickly.

**Tournament selection** fills each parent slot by running a small competition: $k$ individuals are drawn at random from the population, and the one with the best objective value wins the tournament and becomes a parent. Repeating this procedure twice yields a parent pair. The parameter $k$ controls selection pressure: a larger tournament is more elitist, while a tournament of size one is equivalent to uniform random selection.

**Roulette wheel selection** (also called fitness-proportionate selection) assigns each individual a selection probability proportional to its fitness. Fitness is defined relative to the rest of the population:

$$\phi_i = \max_j y^{(j)} - y^{(i)}$$

so that the worst individual has fitness zero and cannot be selected, while the best individual has the highest selection probability. This method applies softer pressure than truncation selection and maintains more diversity, but it can be slow to converge when one individual dominates the rest.

<img width="937" height="671" alt="Selection methods" src="https://github.com/user-attachments/assets/c8f6ebdd-60e4-47b4-8ccc-fc1c00b62776" />

### 1.2.4 Crossover

Crossover combines genetic material from two parents to produce a child. The intention is that the child inherits good traits from both parents. Several schemes are standard:

**Single-point crossover** selects a cut point $i$ uniformly at random from $\{1, \ldots, d-1\}$. The child takes genes $1$ through $i$ from parent A and genes $i+1$ through $d$ from parent B.

<img width="1144" height="182" alt="Single-point crossover" src="https://github.com/user-attachments/assets/16d7a6ce-9859-4630-a6d4-248c47f903b2" />

**Two-point crossover** selects two cut points $i < j$ at random. The child inherits genes $1$ to $i$ from parent A, genes $i+1$ to $j$ from parent B, and genes $j+1$ to $d$ from parent A again. This allows a contiguous block of genetic material from parent B to be transplanted into an otherwise A-majority child.

<img width="1159" height="182" alt="Two-point crossover" src="https://github.com/user-attachments/assets/8b6e3314-d578-4113-8ec1-38cae20772da" />

**Uniform crossover** treats each gene position independently: for each index $i$, the child takes gene $i$ from parent A with probability 0.5, and from parent B otherwise. This maximizes recombination but also disrupts any spatial structure in the chromosome.

<img width="1159" height="159" alt="Uniform crossover" src="https://github.com/user-attachments/assets/c7847f5e-ded9-41cf-8800-16b1ad8193fa" />

For real-valued chromosomes, a natural alternative is **interpolation crossover**: the child is a convex combination of the two parents,

$$\mathbf{x}_\text{child} = (1 - \lambda)\,\mathbf{x}_A + \lambda\,\mathbf{x}_B$$

where $\lambda \in [0,1]$ is a scalar parameter. Setting $\lambda = 0.5$ places the child at the midpoint between the two parents. Other values of $\lambda$ weight the child closer to one parent or the other. Choosing $\lambda$ randomly at each crossover event introduces additional stochasticity.

### 1.2.5 Mutation

Crossover alone cannot introduce genetic material that was absent from the initial population. If the global optimum lies in a region not covered by any individual at initialization, crossover of existing individuals can never reach it. Mutation addresses this by randomly perturbing individuals after crossover, ensuring that new regions of the search space can always be discovered.

For binary chromosomes, each bit is independently flipped with a small probability $\lambda$. A common heuristic is $\lambda = 1/d$, which results in approximately one bit flip per child on average.

For real-valued chromosomes, Gaussian mutation adds zero-mean noise to each gene:

$$x_i \leftarrow x_i + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2)$$

The step size $\sigma$ controls the balance between local refinement (small $\sigma$) and global exploration (large $\sigma$).

<img width="817" height="100" alt="Binary mutation" src="https://github.com/user-attachments/assets/b7352d09-632a-405f-a78c-f39bc2d8e014" />

<img width="1036" height="281" alt="GA on Michalewicz" src="https://github.com/user-attachments/assets/fee23b63-06b4-4126-a636-807314b9fd43" />

The figure above shows a genetic algorithm with truncation selection, single-point crossover, and Gaussian mutation applied to the Michalewicz function, a standard multimodal benchmark.

---

## 1.3 Differential Evolution

Differential evolution (DE) is a population-based method that generates candidate improvements by computing arithmetic differences between individuals. Rather than simulating biological genetics, it exploits the geometric structure of the population: the vector connecting two individuals encodes information about the scale and direction of variation in the objective landscape.

At each iteration, every individual $\mathbf{x}^{(k)}$ is a candidate for replacement. The update proceeds as follows:

1. Select three distinct individuals $\mathbf{a}$, $\mathbf{b}$, $\mathbf{c}$ from the population, all different from $\mathbf{x}^{(k)}$.
2. Compute the trial vector: $\mathbf{z} = \mathbf{a} + w(\mathbf{b} - \mathbf{c})$, where $w \in [0.4, 1.0]$ is the differential weight.
3. Select a random index $j_\text{rand} \in \{1, \ldots, n\}$.
4. Construct the candidate $\mathbf{x}'$ via binary crossover:

$$x'_i = \begin{cases} z_i & \text{if } i = j_\text{rand} \text{ or } U(0,1) < p \\ x_i & \text{otherwise} \end{cases}$$

The index $j_\text{rand}$ guarantees that $\mathbf{x}'$ differs from $\mathbf{x}^{(k)}$ in at least one dimension, preventing the trial vector from being identical to the current individual.

5. Replace $\mathbf{x}^{(k)}$ with $\mathbf{x}'$ if and only if $f(\mathbf{x}') \leq f(\mathbf{x}^{(k)})$.

The perturbation $w(\mathbf{b} - \mathbf{c})$ is adaptive in the sense that its magnitude naturally scales with the spread of the population: as individuals converge, the differences $\mathbf{b} - \mathbf{c}$ shrink, and so does the perturbation. This gives DE an implicit self-adaptation of step size over the course of the run.

<img width="164" height="96" alt="DE mutation illustration" src="https://github.com/user-attachments/assets/1b2554bb-e432-4666-aa26-603c4d87b2dc" />

<img width="1109" height="547" alt="DE on Ackley" src="https://github.com/user-attachments/assets/edcd153f-0d4d-4ebe-83a8-bcec7fcf880f" />

The figure above shows differential evolution with $p = 0.5$ and $w = 0.2$ applied to Ackley's function, a standard multimodal benchmark with a single global minimum at the origin surrounded by many local minima.

---

## 1.4 Particle Swarm Optimization

Particle swarm optimization (PSO) was originally proposed as a model of collective animal behavior, such as the flocking of birds or the schooling of fish. In optimization, each individual is a particle that moves through the design space with a velocity that is updated at every iteration.

Each particle maintains three quantities: its current position $\mathbf{x}^{(i)}$, its current velocity $\mathbf{v}^{(i)}$, and the best position it has personally visited so far, $\mathbf{x}^{(i)}_\text{best}$. The algorithm also tracks $\mathbf{x}_\text{best}$, the best position found by any particle across the entire swarm.

At each iteration, velocities and positions are updated according to:

$$\mathbf{v}^{(i)} \leftarrow w\,\mathbf{v}^{(i)} + c_1\,\mathbf{r}_1 \odot \left(\mathbf{x}^{(i)}_\text{best} - \mathbf{x}^{(i)}\right) + c_2\,\mathbf{r}_2 \odot \left(\mathbf{x}_\text{best} - \mathbf{x}^{(i)}\right)$$

$$\mathbf{x}^{(i)} \leftarrow \mathbf{x}^{(i)} + \mathbf{v}^{(i)}$$

where $\mathbf{r}_1, \mathbf{r}_2 \sim U(\mathbf{0}, \mathbf{1})$ are vectors of independent uniform random numbers drawn fresh at each iteration, and $\odot$ denotes elementwise multiplication.

The three terms in the velocity update have distinct roles. The inertia term $w\,\mathbf{v}^{(i)}$ preserves momentum from the previous iteration, allowing a particle to continue moving in a direction that has been productive. The cognitive term $c_1\,\mathbf{r}_1 \odot (\mathbf{x}^{(i)}_\text{best} - \mathbf{x}^{(i)})$ pulls the particle back toward the best position it has personally experienced. The social term $c_2\,\mathbf{r}_2 \odot (\mathbf{x}_\text{best} - \mathbf{x}^{(i)})$ pulls it toward the best position found by any particle.

A common practical improvement is to let the inertia weight $w$ decay linearly from a value near 1 at the start (favoring exploration) to a value near 0.4 by the end (favoring exploitation). This schedule encourages broad search early in the run and fine-grained convergence later.

PSO converges quickly on unimodal problems because the social term creates strong pull toward the global best. On multimodal problems, early convergence of the swarm to a single region can prevent particles from discovering other basins of attraction, which is why the cognitive term and the randomness in $\mathbf{r}_1$ and $\mathbf{r}_2$ are important for maintaining diversity.

---

## 1.5 Firefly Algorithm

The firefly algorithm draws its inspiration from bioluminescent signaling. Each individual in the population represents a firefly whose brightness corresponds to its fitness: lower objective function value means higher brightness. At each iteration, each firefly is attracted to every other firefly that is brighter than itself, and the strength of attraction decreases with distance.

If firefly $\mathbf{a}$ is less bright than firefly $\mathbf{b}$, then $\mathbf{a}$ moves toward $\mathbf{b}$ according to:

$$\mathbf{a} \leftarrow \mathbf{a} + \beta\,I(\|\mathbf{b} - \mathbf{a}\|)\,(\mathbf{b} - \mathbf{a}) + \alpha\,\boldsymbol{\epsilon}$$

where $\beta$ is the source intensity at zero distance, $I(r)$ is an intensity function that decreases with distance $r$, and $\alpha\,\boldsymbol{\epsilon}$ is a random perturbation with $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.

Three standard choices for $I(r)$ are:

Inverse square law: $I(r) = 1/r^2$, which has a singularity at $r = 0$ and is rarely used in practice.

Exponential decay: $I(r) = e^{-\gamma r}$, which decays monotonically and is well-behaved everywhere.

Gaussian: $I(r) = e^{-\gamma r^2}$, which is the recommended default. It avoids the singularity, decays faster at large distances, and results in smoother movement of the population.

The random walk term $\alpha\,\boldsymbol{\epsilon}$ prevents the algorithm from stagnating when fireflies cluster together and ensures that the population continues to explore. The parameter $\alpha$ should be tuned alongside $\gamma$: large $\alpha$ increases exploration, small $\alpha$ favors local refinement.

A notable property of the firefly algorithm is that the population self-organizes into local clusters, where each cluster is drawn toward a local or global optimum. This makes it naturally suited to multimodal problems, where multiple good solutions may be of interest.

<img width="1109" height="288" alt="Firefly on Branin" src="https://github.com/user-attachments/assets/1cd292ca-f9e6-4359-89c2-519ee50465e9" />

---

## 1.6 Cuckoo Search

Cuckoo search is inspired by the brood parasitism of cuckoo birds, which lay eggs in the nests of other species. The host bird may detect the foreign egg and abandon the nest, or may raise the cuckoo chick unknowingly. This behavior maps to an optimization dynamic where good solutions persist and poor ones are periodically discarded.

In the algorithm, each nest represents a design point with an associated objective function value. At each iteration, two operations take place.

First, a cuckoo lays an egg in a randomly selected host nest. The new design point is generated by a Levy-like flight from the position of another randomly chosen nest:

$$\mathbf{x}_\text{new} = \mathbf{x}_j + \delta, \quad \delta_i \sim \text{Cauchy}(0, \sigma)$$

If the new point has a better objective value than the selected nest, it replaces it. The use of a Cauchy distribution rather than a Gaussian gives the flights a heavy tail, meaning the algorithm occasionally makes very large jumps. This mimics the long-distance foraging patterns observed in animals and allows the algorithm to escape local minima more readily than a Gaussian random walk.

Second, a fraction $p_a$ of the worst nests is abandoned and replaced by new nests generated via Cauchy flights from the surviving nests. This operation plays the role of mutation in genetic algorithms: it injects fresh diversity into the population and prevents premature convergence to a suboptimal region.

The two parameters $p_a$ and the Cauchy scale $\sigma$ control the balance between exploitation (small $p_a$, small $\sigma$) and exploration (large $p_a$, large $\sigma$). A typical value is $p_a = 0.25$.

<img width="1109" height="262" alt="Cuckoo on Branin" src="https://github.com/user-attachments/assets/793d95f5-a10f-4403-9ae3-66863f6bf87a" />

---

## 1.7 Hybrid Methods

Population methods are effective at global search: by maintaining diverse individuals across the design space, they can locate the basin of attraction of the global optimum even in highly multimodal landscapes. However, once a good region has been identified, they converge slowly to the precise optimum within that region. Gradient-based and other local search methods have the opposite property: they converge quickly once started near a good point, but are helpless if that starting point is far from the global optimum.

Hybrid methods, also called memetic algorithms or genetic local search, combine population-based global search with deterministic local search to get the benefits of both. After each round of crossover and mutation, a local optimizer is applied to each individual. The resulting improvement can then be handled in two different ways.

**Lamarckian learning** replaces each individual with its locally optimized version. Both the position and the objective function value that the population sees are those of the improved point. This tends to accelerate convergence because the next generation starts from a more refined set of individuals. The risk is premature convergence: if the local optimizer moves every individual into the same basin of attraction, the population loses diversity and the algorithm cannot escape that region.

**Baldwinian learning** applies the same local optimization, but does not modify the actual position of any individual. Instead, the locally optimized objective value is used as a proxy for the individual's fitness during selection, while the individual itself remains at its original location. This allows the algorithm to bias selection toward individuals that, if locally optimized, would yield good results, without actually committing to those local optima. Diversity is preserved, and the population retains the ability to escape to better regions in subsequent iterations.

<img width="901" height="720" alt="Lamarckian vs Baldwinian" src="https://github.com/user-attachments/assets/db6c1918-e46f-4f79-a16e-e37dba81b2a9" />

The figure illustrates the difference on a one-dimensional function $f(x) = -e^{-x^2} - 2e^{-(x-3)^2}$, which has a local minimum near $x = 0$ and a global minimum near $x = 3$. A population initialized near $x = 0$ under Lamarckian learning is pulled into the local minimum and becomes trapped there. Under Baldwinian learning, the positions remain near $x = 0$ while the perceived fitness encourages selection of individuals that could potentially reach $x = 3$ in future iterations.

---

## 1.8 Summary

Population methods form a broad and practically important class of optimization algorithms with a common structure: maintain a diverse collection of candidate solutions, evaluate their fitness, and use the results to guide the generation of better candidates in the next iteration.

Genetic algorithms model this process on Darwinian evolution, using selection, crossover, and mutation as their core operators. Differential evolution exploits vector differences within the population to generate perturbations that naturally adapt to the current spread of individuals. Particle swarm optimization adds momentum to each individual, creating a swarm that collectively moves toward the best known positions while retaining stochastic diversity. The firefly algorithm and cuckoo search draw on different natural metaphors but share the same essential structure of local attraction and global exploration.

All of these methods can be extended with local search components. Lamarckian learning replaces individuals with their locally improved versions, while Baldwinian learning uses local improvement only to inform selection without modifying the population itself.

The choice among these methods depends on the problem at hand. For smooth but multimodal functions, differential evolution and PSO tend to perform well. For problems where evaluation is cheap and the landscape is highly irregular, genetic algorithms with large populations are a reliable choice. When the optimum may be far from any reasonable prior, Cauchy-based methods such as cuckoo search offer an advantage through their heavy-tailed exploration.

---

**Reference:** Kochenderfer, M. J. and Wheeler, T. A. (2019). *Algorithms for Optimization*. MIT Press.

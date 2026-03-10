---
layout: default
---
This repository describes the theory behind "Populations Methods" and after that, it try to use this methdos in prartical projects

The theory part, is from chapter 9 of Kochenderfer, M. J. & Wheeler, T. A. (2019). Algorithms for Optimization. The MIT Press. The various sections have been modified with other materials found on the web

# 1. Theory: Population Methods

population methods involve optimization using a collection of design points, called **individuals**. Having a large number of individuals distributed throughout the design space can help the algorithm avoid becoming stuck in a local minimum (this is an improvement from methods  where a single design point is moved incrementally toward a minimum). 

Information at different points in the design space can be shared between individuals to globally optimize the objective function. Most population methods are stochastic in nature, and it is generally easy to parallelize the computation.

Unlike "trajectory-based" methods (like Gradient Descent) that follow a single path, these algorithms throw a whole crowd of potential solutions into the search space and let them interact to find the best answer.

It’s essentially "survival of the fittest" applied to data.

Why we don't just use standard calculus (gradients), it’s because real-world math is often messy. Population methods offer:

- Global Exploration: They are less likely to get stuck in a "local trap" (a small hill that looks like a mountain until you see the real Everest further away).

- No Gradient Needed: They work on "black box" problems where we don't have a neat formula for the derivative.

- Parallelism: Since you have many candidates, you can calculate their fitness simultaneously on multiple processors.

Note: The trade-off is speed. These methods usually require more "fitness evaluations" (calculations) than gradient-based methods.
---

## 1.1 Initialization

Population methods begin with an initial population, just as descent methods require an initial design point. The initial population should be spread over the design space to increase the chances that the samples are close to the best regions.

We can often constrain the design variables to a region of interest consisting of a hyperrectangle defined by lower and upper bounds **a** and **b**. Initial populations can be sampled from a uniform distribution for each coordinate:

$$x_i^{(j)} \sim U(a_i, b_i)$$

where $x^{(j)}$ is the $j$th individual in the population.

<img width="1080" height="594" alt="Screenshot 2026-02-25 alle 17 04 49" src="https://github.com/user-attachments/assets/445d7337-4bf6-46d9-a241-5cdb46c51699" />

A comparison of the normal distribution with standard deviation 1 and the Cauchy dis- tribution with scale 1. Although σ is sometimes used for the scale parameter in the Cauchy distri- bution, this should not be con- fused with the standard deviation since the standard deviation of the Cauchy distribution is undefined. The Cauchy distribution is heavy- tailed, allowing it to cover the de- sign space more broadly.


<img width="1236" height="583" alt="Screenshot 2026-02-25 alle 17 07 48" src="https://github.com/user-attachments/assets/50a6c216-561a-4efe-a6bc-22cafcc5adc7" />

nota: va messo riformattato emesso le formule
A comparison of the normal distribution with standard deviation 1 and the Cauchy dis- tribution with scale 1. Although σ is sometimes used for the scale parameter in the Cauchy distri- bution, this should not be con- fused with the standard deviation since the standard deviation of the Cauchy distribution is undefined. The Cauchy distribution is heavy- tailed, allowing it to cover the de- sign space more broadly.
---

## 1.2 Genetic Algorithms

**Genetic algorithms** borrow inspiration from biological evolution, where fitter individuals are more likely to pass on their genes to the next generation. An individual's fitness for reproduction is inversely related to the value of the objective function at that point. The design point associated with an individual is represented as a **chromosome**. At each generation, the chromosomes of the fitter individuals are passed on to the next generation after undergoing the genetic operations of **crossover** and **mutation**.


### 1.2.1 Chromosomes

There are several ways to represent chromosomes. The simplest is the **binary string chromosome**, a representation similar to the way DNA is encoded. A random binary string of length `d` can be generated using `bitrand(d)`.

Binary strings are often used due to the ease of expressing crossover and mutation. It is often more natural to represent a chromosome using a list of real values. Such **real-valued chromosomes** are vectors in $\mathbb{R}^d$ that directly correspond to points in the design space.

<img width="1150" height="122" alt="Screenshot 2026-02-25 alle 17 06 03" src="https://github.com/user-attachments/assets/7f2b356a-10f5-490e-b7bb-728623ba7974" />

A chromosome represented as a binary string.

### 1.2.2 Initialization

Genetic algorithms start with a random initial population. Binary string chromosomes are typically initialized using random bit strings.


### 1.2.3 Selection

Selection is the process of choosing chromosomes to use as parents for the next generation. For a population with `m` chromosomes, a selection method will produce a list of `m` parental pairs for the `m` children of the next generation.

There are several approaches for biasing the selection toward the fittest:

- **Truncation selection** — sample parents from among the best `k` chromosomes in the population.
- **Tournament selection** — each parent is the fittest out of `k` randomly chosen chromosomes.
- **Roulette wheel selection** (fitness proportionate selection) — each parent is chosen with a probability proportional to its performance relative to the population. The fitness of individual $i$ is assigned according to $\max\{y^{(1)}, \ldots, y^{(m)}\} - y^{(i)}$.

<img width="937" height="671" alt="Screenshot 2026-02-25 alle 17 09 26" src="https://github.com/user-attachments/assets/c8f6ebdd-60e4-47b4-8ccc-fc1c00b62776" />



### 1.2.4 Crossover

Crossover combines the chromosomes of parents to form children. There are several crossover schemes:

- **Single-point crossover** — the first portion of parent A's chromosome forms the first portion of the child, and the latter portion of parent B's chromosome forms the latter part. The crossover point is determined uniformly at random.

<img width="1144" height="182" alt="Screenshot 2026-02-25 alle 17 11 55" src="https://github.com/user-attachments/assets/16d7a6ce-9859-4630-a6d4-248c47f903b2" />

- **Two-point crossover** — uses two random crossover points.
<img width="1159" height="182" alt="Screenshot 2026-02-25 alle 17 12 33" src="https://github.com/user-attachments/assets/8b6e3314-d578-4113-8ec1-38cae20772da" />


- **Uniform crossover** — each bit has a fifty percent chance of coming from either parent.

<img width="1159" height="159" alt="Screenshot 2026-02-25 alle 17 13 03" src="https://github.com/user-attachments/assets/c7847f5e-ded9-41cf-8800-16b1ad8193fa" />


For real-valued chromosomes, values can also be linearly interpolated between the parents:

$$x \leftarrow (1 - \lambda)x_a + \lambda x_b$$

where $\lambda$ is a scalar parameter typically set to one-half.




### 1.2.5 Mutation

If new chromosomes were produced only through crossover, many traits not present in the initial random population could never occur, and the most-fit genes could saturate the population. **Mutation** allows new traits to spontaneously appear, allowing the genetic algorithm to explore more of the state space. Child chromosomes undergo mutation after crossover.

Each bit in a binary-valued chromosome typically has a small probability of being flipped. For a chromosome with `m` bits, this mutation rate is typically set to `1/m`, yielding an average of one mutation per child chromosome. Mutation for real-valued chromosomes is more commonly implemented by adding zero-mean Gaussian noise.

<img width="817" height="100" alt="Screenshot 2026-02-25 alle 17 16 38" src="https://github.com/user-attachments/assets/b7352d09-632a-405f-a78c-f39bc2d8e014" />
above44
Mutation for binary string chromosomes gives each bit a small probability of flipping.



<img width="1036" height="281" alt="Screenshot 2026-02-25 alle 17 14 02" src="https://github.com/user-attachments/assets/fee23b63-06b4-4126-a636-807314b9fd43" />
above
A genetic algorithm with truncation selection, single point crossover, and Gaussian mu- tation with σ = 0.1 applied to the Michalewicz function

---

## 1.3 Differential Evolution

**Differential evolution** attempts to improve each individual in the population by recombining other individuals according to a simple formula. It is parameterized by a crossover probability `p` and a differential weight `w`. Typically, `w` is between 0.4 and 1. For each individual `x`:

1. Choose three random distinct individuals **a**, **b**, and **c**.
2. Construct an interim design $z = a + w \cdot (b - c)$.
3. Choose a random dimension $j \in [1, \ldots, n]$.
4. Construct the candidate individual $x'$ using binary crossover:

$$x'_i = \begin{cases} z_i & \text{if } i = j \text{ or with probability } p \\ x_i & \text{otherwise} \end{cases}$$

5. Insert the better design between $x$ and $x'$ into the next generation.


Differential evolution takes three individuals a, b, and c and combines them to form the can- didate individual z. watch:


<img width="164" height="96" alt="Screenshot 2026-02-25 alle 17 18 00" src="https://github.com/user-attachments/assets/1b2554bb-e432-4666-aa26-603c4d87b2dc" />

The algorithm is demonstrated with the following image. in particular you can see:
Differential evolution with p = 0.5 and w = 0.2 applied to Ackley’s function,
<img width="1109" height="547" alt="Screenshot 2026-02-25 alle 17 19 00" src="https://github.com/user-attachments/assets/edcd153f-0d4d-4ebe-83a8-bcec7fcf880f" />

---

## 1.4 Particle Swarm Optimization

**Particle swarm optimization** introduces momentum to accelerate convergence toward minima. Each individual, or *particle*, in the population keeps track of its current position, velocity, and the best position it has seen so far. Momentum allows an individual to accumulate speed in a favorable direction, independent of local perturbations.


At each iteration, each individual is accelerated toward both the best position it has seen and the best position found thus far by any individual. The update equations are:

$$x^{(i)} \leftarrow x^{(i)} + v^{(i)}$$

$$v^{(i)} \leftarrow wv^{(i)} + c_1 r_1 \left(x^{(i)}_\text{best} - x^{(i)}\right) + c_2 r_2 \left(x_\text{best} - x^{(i)}\right)$$

where $x_\text{best}$ is the best location found so far over all particles; $w$, $c_1$, and $c_2$ are parameters; and $r_1$ and $r_2$ are random numbers drawn from $U(0, 1)$.

> A common strategy is to allow the inertia $w$ to decay over time.

---

## 1.5 Firefly Algorithm

The **firefly algorithm** was inspired by the manner in which fireflies flash their lights to attract mates. In the firefly algorithm, each individual in the population is a firefly and can flash to attract other fireflies. At each iteration, all fireflies are moved toward all more attractive fireflies. A firefly **a** is moved toward a firefly **b** with greater attraction according to:

$$a \leftarrow a + \beta I(\|b - a\|)(b - a) + \alpha \epsilon$$

where $I$ is the intensity of the attraction and $\beta$ is the source intensity. A random walk component is included as well, where $\epsilon$ is drawn from a zero-mean, unit covariance multivariate Gaussian, and $\alpha$ scales the step size.

The intensity $I$ decreases as the distance $r$ between the two fireflies increases and is defined to be 1 when $r = 0$. Several models are available:

- **Inverse square law**: $I(r) = \dfrac{1}{r^2}$

- **Exponential decay** (absorbed in medium): $I(r) = e^{-\gamma r}$

- **Gaussian drop-off** (recommended — avoids singularity at $r = 0$): $I(r) = e^{-\gamma r^2}$


Firefly search, with α = 0.5,β = 1,andγ = 0.1ap- plied to the Branin function
<img width="1109" height="288" alt="Screenshot 2026-02-25 alle 17 20 39" src="https://github.com/user-attachments/assets/1cd292ca-f9e6-4359-89c2-519ee50465e9" />


---

## 1.6 Cuckoo Search

**Cuckoo search** is another nature-inspired algorithm named after the cuckoo bird, which engages in a form of brood parasitism. Cuckoos lay their eggs in the nests of other birds; the host bird may detect the invasive egg and destroy it or establish a new nest elsewhere, or may accept and raise it.

In cuckoo search, each nest represents a design point. New design points are produced using **Lévy flights** from nests — random walks with step-lengths from a heavy-tailed distribution. A new design point can replace a nest if it has a better objective function value.

The core rules are:

1. A cuckoo will lay an egg in a randomly chosen nest.
2. The best nests with the best eggs will survive to the next generation.
3. Cuckoo eggs have a chance of being discovered by the host bird, in which case the eggs are destroyed.

Cuckoo search uses a **Cauchy distribution** for random flights, which has a heavier tail than uniform or Gaussian distributions and is more representative of animal movement patterns in the wild.

> Other nature-inspired algorithms include the artificial bee colony, the gray wolf optimizer, the bat algorithm, glowworm swarm optimization, intelligent water drops, and harmony search. There has been some criticism of the proliferation of methods that make analogies to nature without fundamentally contributing novel methods and understanding.

---

## 1.7 Hybrid Methods

Many population methods perform well in **global search**, being able to avoid local minima and finding the best regions of the design space. However, these methods do not perform as well in **local search** compared to descent methods. Several hybrid methods (also referred to as *memetic algorithms* or *genetic local search*) have been developed to extend population methods with descent-based features.

Cuckoo search applied to the Branin function

<img width="1109" height="262" alt="Screenshot 2026-02-25 alle 17 21 40" src="https://github.com/user-attachments/assets/793d95f5-a10f-4403-9ae3-66863f6bf87a" />


There are two general approaches:

- **Lamarckian learning** — the population method is extended with a local search method that locally improves each individual. The original individual and its objective function value are **replaced** by the individual's optimized counterpart.

- **Baldwinian learning** — the same local search method is applied to each individual, but the results are used only to update the individual's **perceived** objective function value. Individuals are not replaced but are merely associated with optimized objective function values. Baldwinian learning can help prevent premature convergence.

### Example — Lamarckian vs. Baldwinian Learning

<img width="901" height="720" alt="Screenshot 2026-02-25 alle 17 23 08" src="https://github.com/user-attachments/assets/db6c1918-e46f-4f79-a16e-e37dba81b2a9" />


Consider optimizing $f(x) = -e^{-x^2} - 2e^{-(x-3)^2}$ using a population of individuals initialized near $x = 0$.

A **Lamarckian** local search update applied to this population would move the individuals toward the local minimum, reducing the chance that future individuals escape and find the global optimum near $x = 3$.

A **Baldwinian** approach will compute the same update but leaves the original designs unchanged. The selection step will value each design according to its value from a local search, preserving diversity and improving the chances of finding the global optimum.

---

## 1.8 Summary

- Population methods use a collection of individuals in the design space to guide progression toward an optimum.
- Genetic algorithms leverage **selection**, **crossover**, and **mutations** to produce better subsequent generations.
- **Differential evolution**, **particle swarm optimization**, the **firefly algorithm**, and **cuckoo search** include rules and mechanisms for attracting design points to the best individuals in the population while maintaining suitable state space exploration.
- Population methods can be extended with local search approaches to improve convergence.



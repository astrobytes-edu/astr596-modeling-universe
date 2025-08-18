ASTR 596: Modeling the Universe
Weeks 3 & 4: Dynamics and Statistical Sampling
🎯 This Week's Goal: From Clockwork to Chaos
In our first project, we treated stars as isolated, static objects defined by their intrinsic properties. But the universe is a dynamic, chaotic dance floor, governed by the relentless pull of gravity. This week, we breathe life into our stellar populations by simulating their gravitational interactions over cosmic timescales. In doing so, we'll confront a fundamental truth of physics: even with simple, deterministic rules, the collective behavior of many interacting bodies often becomes impossible to solve perfectly. This complexity is not a bug; it's the feature that gives rise to the rich structures we observe, from stable solar systems to majestic star clusters.

This journey leads us to two core pillars of computational science:

Numerical Integration: How do we approximate the solution to problems we can't solve analytically? You'll build the engine for your N-body simulator from the ground up. This isn't just about coding equations; it's about developing an intuition for numerical stability, error propagation, and energy conservation. You'll discover firsthand why the choice of algorithm is a critical decision that directly impacts the physical realism and long-term validity of your simulation.

Monte Carlo Methods: How do we create a meaningful starting point for our simulation? A realistic model requires realistic initial conditions. We will harness the power of structured randomness through Monte Carlo techniques. You'll learn how to sample from complex probability distributions to assign stellar masses and arrange them into a gravitationally bound cluster, transforming abstract statistical descriptions into a concrete, simulated reality.

By the end of this block, you will have built a working simulation of a star cluster, a foundational tool in computational astrophysics that serves as a laboratory for exploring stellar evolution, galactic dynamics, and the very formation of structure in the cosmos.

Part 1: N-Body Dynamics - The Clockwork Universe
The N-body problem is deceptively simple to state: given N objects with known initial positions and velocities, predict their future motion under their mutual gravitational attraction. For N=2, Newton provided a perfect, elegant analytical solution—the conic sections of Kepler's orbits that describe planets moving around a star. For N≥3, however, the intricate web of interactions introduces chaos, and the problem has no general analytical solution. This famous "three-body problem" has challenged mathematicians and physicists for centuries and marks the boundary where we must transition from pure theory to computational approximation.

🔬 The Physics: Newton's Law of Gravitation
The foundation of our simulation is Newton's law of universal gravitation. The force F 
ij
​
  exerted on a particle i by a particle j is a vector quantity:

F 
ij
​
 =G 
∣r 
ij
​
 ∣ 
3
 
m 
i
​
 m 
j
​
 
​
 r 
ij
​
 
where:

G is the gravitational constant, a fundamental constant of nature.

m 
i
​
  and m 
j
​
  are the masses of the two particles.

r 
ij
​
 =r 
j
​
 −r 
i
​
  is the displacement vector pointing from particle i to particle j. The magnitude ∣r 
ij
​
 ∣ is the distance between them. The term r 
ij
​
 /∣r 
ij
​
 ∣ is a unit vector that gives the force its direction.

The total, or net, force on particle i is the vector sum of the forces from all other particles in the system. This is the principle of superposition.

F 
i
​
 = 
j

=i
∑
N
​
 F 
ij
​
 
From Newton's second law, F 
i
​
 =m 
i
​
 a 
i
​
 , we can find the acceleration of particle i by dividing the total force by its own mass:

a 
i
​
 = 
dt 
2
 
d 
2
 r 
i
​
 
​
 = 
m 
i
​
 
F 
i
​
 
​
 = 
j

=i
∑
N
​
 G 
∣r 
ij
​
 ∣ 
3
 
m 
j
​
 
​
 r 
ij
​
 
This gives us a system of N coupled second-order ordinary differential equations (ODEs). "Coupled" is the key word here; the acceleration of every particle depends on the instantaneous position of every other particle. You cannot solve for the motion of one star without simultaneously solving for all the others. Our task is to develop a computational framework to solve this intricate system.

Part 2: Solving the Unsolvable - Numerical Integration of ODEs
Since we can't solve these ODEs with a pen and paper, we turn to computers. Numerical integration involves breaking the smooth, continuous flow of time into a series of discrete steps of size Δt. We use the system's state (all positions and velocities) at a given moment to approximate its state a short time later. The accuracy and stability of our simulation depend entirely on how we perform this approximation.

To make the problem more tractable for standard solvers, we rewrite our single second-order ODE for each particle as a pair of two first-order ODEs:

dt
dr 
i
​
 
​
 =v 
i
​
 
dt
dv 
i
​
 
​
 =a 
i
​
 (r 
1
​
 ,…,r 
N
​
 )
Let's denote the state at time t 
n
​
  as (r 
n
​
 ,v 
n
​
 ). Our challenge is to find the state at the next time step, t 
n+1
​
 =t 
n
​
 +Δt.

Method 1: Euler's Method (The "Don't Use This" Method)
The most straightforward approach is to assume that the velocity and acceleration are constant across the entire time step Δt. This is equivalent to taking the first term of a Taylor series expansion.

r 
n+1
​
 =r 
n
​
 +v 
n
​
 Δt
v 
n+1
​
 =v 
n
​
 +a 
n
​
 Δt
Pros: Incredibly simple to understand and implement. It serves as an excellent conceptual starting point and a baseline for appreciating more sophisticated methods.

Cons: It is numerically unstable and physically inaccurate for orbital mechanics. Because it only uses information at the beginning of the step, it consistently overestimates the curvature of an orbit, causing the simulated object to gain energy with every step. For an orbit, this means the planet will unnaturally spiral outwards, violating the law of conservation of energy. It's a great first step for learning, but terrible for any real physics.

Method 2: Runge-Kutta 4th Order (RK4) (The "Workhorse")
RK4 is a much more intelligent and accurate method. Instead of taking one blind step based on the gradient (the forces) at the start of the interval, it cleverly probes the gradient at four different points within the time step and combines them in a weighted average. It's like "peeking ahead" to see how the forces are changing and adjusting its trajectory accordingly.

For a general ODE dy/dt=f(t,y), the update is:

y 
n+1
​
 =y 
n
​
 + 
6
1
​
 (k 
1
​
 +2k 
2
​
 +2k 
3
​
 +k 
4
​
 )Δt
where:

k 
1
​
 =f(t 
n
​
 ,y 
n
​
 ) (The simple Euler step)

k 
2
​
 =f(t 
n
​
 + 
2
Δt
​
 ,y 
n
​
 + 
2
k 
1
​
 Δt
​
 ) (Step to the midpoint using the initial slope k 
1
​
 )

k 
3
​
 =f(t 
n
​
 + 
2
Δt
​
 ,y 
n
​
 + 
2
k 
2
​
 Δt
​
 ) (Another step to the midpoint, but using the more accurate slope k 
2
​
 )

k 
4
​
 =f(t 
n
​
 +Δt,y 
n
​
 +k 
3
​
 Δt) (Step all the way to the end using the refined midpoint slope k 
3
​
 )

Pros: Very accurate and stable for a wide variety of problems, not just gravity. It's a fantastic general-purpose ODE solver.

Cons: It requires four full force evaluations per time step, making it computationally four times more expensive than Euler. While it conserves energy much better than Euler, it is not perfectly energy-conserving and will show a slow, random drift in total energy over very long timescales.

Method 3: Leapfrog (Verlet) Integration (The "Astrophysicist's Choice")
Leapfrog is a special type of integrator known as a symplectic integrator. These methods are specifically designed for Hamiltonian systems (like gravity), where the conservation of physical quantities like energy and momentum is paramount.

The clever trick is to evaluate positions and velocities at "staggered" time steps that leapfrog over each other. Velocities are calculated at the half-step (t 
n−1/2
​
 ,t 
n+1/2
​
 ), while positions are calculated at the full step (t 
n
​
 ,t 
n+1
​
 ). This seemingly small change has profound consequences for the long-term stability of the simulation.

The update equations are executed in a sequence:

Kick: Update velocity from t 
n−1/2
​
  to t 
n+1/2
​
  using the acceleration at the current position t 
n
​
 .

v 
n+1/2
​
 =v 
n−1/2
​
 +a 
n
​
 Δt
Drift: Update position from t 
n
​
  to t 
n+1
​
  using this newly calculated half-step velocity.

r 
n+1
​
 =r 
n
​
 +v 
n+1/2
​
 Δt
This "kick-drift-kick" sequence gives the method its name and its power.

Pros: Excellent long-term energy conservation. While the total energy will oscillate slightly around the true value during an orbit, it does not systematically drift over time. This makes it the gold standard for gravitational simulations. It is also time-reversible, a key property of physical laws.

Cons: It is formally less accurate than RK4 for a single step of a given size, but its superior properties make it far more reliable for long-term orbital dynamics.

Energy Conservation as a Diagnostic Tool
In any closed physical system governed by conservative forces like gravity, the total energy must remain constant. This is a fundamental law of physics. While our numerical methods are only approximations, how well they respect this law is the single most important test of their validity.

Defining the System's Energy
The total energy E of the N-body system is the sum of its total kinetic energy T and its total potential energy U.

Kinetic Energy (T): The energy of motion. For a system of N particles, it is the sum of the kinetic energies of each particle:

T= 
i=1
∑
N
​
  
2
1
​
 m 
i
​
 ∣v 
i
​
 ∣ 
2
 
Potential Energy (U): The energy stored in the gravitational field. It is the sum of the potential energies of every unique pair of particles in the system:

U= 
i<j
∑
N
​
 −G 
∣r 
ij
​
 ∣
m 
i
​
 m 
j
​
 
​
 
The sum is over i<j to ensure we only count each pair once.

The total energy is therefore E=T+U. In a perfect simulation, E(t) would be a flat line. In reality, numerical errors cause it to change.

How Energy Evolves for Different Integrators
Euler: Shows a systematic, linear drift. Energy is constantly added to the system, causing orbits to spiral outwards unphysically.

RK4: Shows a "random walk" in energy. The error in each step is random, so the total energy drifts slowly away from the true value over many steps.

Leapfrog: Shows bounded oscillations. The energy fluctuates around the true value but does not systematically drift away. This is because symplectic integrators conserve a "shadow Hamiltonian" that is very close to the true Hamiltonian of the system, ensuring long-term stability.

Monitoring the fractional energy error, ∣ΔE/E 
0
​
 ∣=∣(E(t)−E 
0
​
 )/E 
0
​
 ∣, is our primary tool for debugging and validation. A large, sudden jump indicates a problem (like a dangerously close encounter between two stars), while a steady drift tells us our timestep Δt is too large.

Adaptive Timestepping
A fixed timestep Δt is inefficient. During a close encounter, forces change rapidly, requiring a very small Δt to maintain accuracy. When stars are far apart, they move slowly, and a much larger Δt would suffice, saving computational time.

Adaptive timestepping is the solution: we adjust Δt on the fly based on the state of the system. A simple and robust way to do this is to use our energy conservation check.

Set a Tolerance: Define a maximum acceptable fractional energy change per step, tol (e.g., 10 
−6
 ).

Take a Trial Step: Advance the system by one timestep Δt.

Check the Error: Calculate the change in total energy, ∣ΔE/E∣.

Accept or Reject:

If ∣ΔE/E∣>tol, the step was inaccurate. Reject it. Restore the system to its previous state, reduce the timestep (e.g., Δt 
new
​
 =0.9Δt(tol/∣ΔE/E∣) 
1/p
  where p is the order of the integrator), and try the step again.

If ∣ΔE/E∣≤tol, the step was accurate. Accept it. You can use the same formula to potentially increase the next timestep for efficiency, up to a maximum allowed value.

This ensures that the simulation "slows down" during complex interactions and "speeds up" during quiet periods, maximizing both accuracy and efficiency.

[Image comparing Euler, RK4, and Leapfrog orbits over time]

Part 3: The Power of Randomness - Monte Carlo Methods
To simulate a realistic star cluster, we can't just place stars on a regular grid or give them all the same mass. We need to initialize their properties (masses, positions, velocities) by sampling from probability distributions that reflect nature. This brings us to the realm of Monte Carlo methods, a class of algorithms that rely on repeated random sampling to obtain numerical results.

The Core Idea: Using Randomness to Find Certainty
The philosophy behind Monte Carlo methods is to solve deterministic problems using probability. Imagine trying to find the area of a complex shape, like a lake on a map. You could overlay a square grid and painstakingly count the cells, but a Monte Carlo approach is simpler: randomly throw a large number of pebbles into a square area that contains the lake. The ratio of pebbles that land in the lake to the total number of pebbles thrown, multiplied by the area of the square, gives you an estimate of the lake's area. As you throw more pebbles, this estimate converges to the true area.

Mathematically, this is an application of the Law of Large Numbers. It allows us to approximate the value of a complex integral by taking the average of a function evaluated at many random points drawn from a distribution. The expectation of a function f(x) where x is drawn from a probability distribution p(x) is:

E[f(x)]=∫f(x)p(x)dx
We can approximate this integral by drawing N samples x 
i
​
  from p(x) and calculating the mean:

E[f(x)]≈ 
N
1
​
  
i=1
∑
N
​
 f(x 
i
​
 )
This powerful idea allows us to build complex systems from simple statistical rules, which is exactly what we need to do to initialize our star cluster.

Sampling the Initial Mass Function (IMF)
Stars are not born with equal masses; nature has a strong preference for creating small stars. The Initial Mass Function (IMF) is a probability distribution that describes the distribution of stellar masses at the time of formation.

The Kroupa IMF (2001)
Modern observations have shown that a single power law is too simplistic. The Kroupa IMF is a more accurate, segmented or "broken" power law that uses different exponents for different mass ranges:

ξ(m)∝ 
⎩
⎨
⎧
​
  
m 
−0.3
 
m 
−1.3
 
m 
−2.3
 
​
  
if 0.01≤m/M 
⊙
​
 <0.08 (Brown Dwarfs)
if 0.08≤m/M 
⊙
​
 <0.5 (Low-mass Stars)
if 0.5≤m/M 
⊙
​
  (High-mass Stars)
​
 
This form reflects our understanding that the physics of star formation may differ for objects of different masses. The flattening of the slope at low masses indicates that while small stars are common, the universe doesn't produce an infinite number of tiny brown dwarfs.

Sampling Method: Inverse Transform Sampling
To computationally sample from these distributions, we can use a powerful technique called Inverse Transform Sampling:

Find the Cumulative Distribution Function (CDF): The PDF, p(m)=C⋅ξ(m), tells us the relative probability of a given mass (where C is a normalization constant). The CDF, P(M<m)=∫ 
m 
min
​
 
m
​
 p(m 
′
 )dm 
′
 , tells us the total probability of finding a star with a mass less than m. For a broken power law like Kroupa's, this involves integrating each segment separately and stitching them together.

Invert the CDF: We solve the CDF equation for the mass m as a function of the probability, giving us m(P). This function takes a probability (a number between 0 and 1) and returns the corresponding mass.

Sample: We generate a random number u from a uniform distribution between [0,1]. Plugging this into our inverted CDF, the result, m(u), is a random mass that is guaranteed to be drawn from the correct distribution.

Sampling Spatial Distributions: The King and Plummer Profiles
Star clusters are not uniformly dense spheres; they are highly concentrated at the center and become progressively sparser outwards. Several models exist to describe this structure.

The King Profile
The King Profile is a classic empirical model that accurately describes the surface brightness (and by extension, stellar density ρ) as a function of radius r from the cluster center. It is derived from a simplified model of stellar dynamics and is defined by a flat central core that transitions to a power-law decline.

The density ρ(r) is given by:

ρ(r)=ρ 
0
​
 [(1+ 
r 
c
2
​
 
r 
2
 
​
 ) 
−1/2
 −(1+ 
r 
c
2
​
 
r 
t
2
​
 
​
 ) 
−1/2
 ] 
2
 
ρ 
0
​
  is the central density of the cluster.

r 
c
​
  is the core radius, defining the size of the flat central region.

r 
t
​
  is the tidal radius, where the cluster's density drops to zero, representing the edge of the cluster as set by the host galaxy's tidal forces.

This equation provides a physically motivated and observationally confirmed profile, but its complexity makes it difficult to work with analytically.

The Plummer Profile
A simpler, fully analytical model often used for theoretical studies is the Plummer sphere. It also describes a centrally concentrated spherical system but has a more convenient mathematical form.

The density ρ(r) for a Plummer sphere is given by:

ρ(r)= 
4πa 
3
 
3M
​
 (1+ 
a 
2
 
r 
2
 
​
 ) 
−5/2
 
M is the total mass of the cluster.

a is the Plummer radius or scale length, which sets the characteristic size of the cluster core.

While less physically detailed than the King model (it lacks a distinct tidal cutoff), the Plummer profile's mathematical simplicity makes it extremely useful for initializing N-body simulations where an analytical potential and density are advantageous.

Sampling Method: Rejection Sampling
To generate 3D positions according to these complex profiles, a direct approach like inverse transform sampling is difficult. Instead, we use a clever and versatile algorithm called Rejection Sampling:

Define a Bounding Box: Enclose the desired volume (e.g., a sphere of radius R 
max
​
 ) in a simple shape we can easily sample from, like a cube. Find the maximum density, ρ 
max
​
  (which occurs at the center, r=0).

Generate a Trial Point: Pick a random 3D point (x,y,z) uniformly within the bounding box.

Calculate Target Density: Evaluate the density profile (either King or Plummer) ρ(r) at that point, where r= 
x 
2
 +y 
2
 +z 
2
 
​
 .

Generate a Random Height: Pick another random number h uniformly between [0,ρ 
max
​
 ]. This represents a random "height" on our density plot.

Accept or Reject: If the random height h is less than the actual density ρ(r) at our trial point (i.e., if our random point falls "under the curve" of the density profile), we accept the point. If not, we reject it and go back to step 2.

The collection of accepted points will have a spatial distribution that perfectly matches the chosen profile, creating a realistic, centrally-condensed star cluster.

💻 Lab Goals for Weeks 3 & 4
Build the Dynamics Engine: Implement the gravitational force calculation for an N-body system. Pay attention to vectorization with NumPy to make your code efficient.

Implement the Integrators: Code the Euler, RK4, and Leapfrog methods. Validate them on a simple, stable two-body problem. Track the total energy of the system at each step and plot its fractional error over time to verify the properties of each integrator.

Implement Adaptive Timestepping: Implement a simple adaptive timestepping scheme for your RK4 or Leapfrog integrator based on an energy conservation tolerance.

Implement the Samplers: Write Python functions to sample stellar masses from the Kroupa IMF using inverse transform sampling and to sample 3D positions from a King or Plummer profile using rejection sampling.

Put It All Together: Combine all your components. Use your samplers to generate a realistic set of initial conditions (masses and positions) for ~100 stars. Use the Leapfrog integrator with adaptive timestepping to simulate the cluster's evolution. Create a 3D visualization of your initial star cluster.
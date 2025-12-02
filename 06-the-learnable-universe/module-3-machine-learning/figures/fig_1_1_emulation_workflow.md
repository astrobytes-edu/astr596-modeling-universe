# Figure 1.1: Emulation Workflow

**Location**: After line 35 in Part I (Introduction)
**Pedagogical Goal**: Provide high-level overview before diving into mathematical details

```mermaid
flowchart TD
    %% Main workflow
    A[("Expensive Simulator<br/>(N-body, Hydro, etc.)")] -->|"Run at<br/>design points"| B["Training Data<br/>{(θ₁, y₁), ..., (θₙ, yₙ)}"]
    B -->|"Learn mapping<br/>θ → y"| C["GP Emulator<br/>p(f | data)"]
    C -->|"Instant<br/>predictions"| D["Fast Predictions<br/>at new θ*"]

    %% Annotations
    B -.->|"Smart design:<br/>Latin Hypercube,<br/>Sobol, etc."| E["Design Points<br/>θ ∈ ℝᵈ"]
    C -.->|"Uncertainty<br/>quantification"| F["μ(θ*) ± σ(θ*)"]
    D -.->|"Enable"| G["Parameter Inference<br/>MCMC, Nested Sampling"]
    D -.->|"Enable"| H["Sensitivity Analysis<br/>Which params matter?"]

    %% Styling
    classDef expensive fill:#ffcccc,stroke:#cc0000,stroke-width:3px
    classDef training fill:#cce5ff,stroke:#0066cc,stroke-width:2px
    classDef gp fill:#ccffcc,stroke:#00cc00,stroke-width:3px
    classDef fast fill:#ffffcc,stroke:#cccc00,stroke-width:2px
    classDef annotation fill:#f0f0f0,stroke:#666666,stroke-dasharray: 5 5

    class A expensive
    class B training
    class C gp
    class D fast
    class E,F,G,H annotation
```

## Key Insight

**The Problem**: Running simulator 10⁶ times for MCMC is impossible (years of compute)

**The Solution**: Run simulator ~100 times → Train GP → Query GP 10⁶ times in seconds

**The Magic**: GP provides both prediction μ(θ) AND uncertainty σ(θ), enabling:
- Trustworthy predictions with error bars
- Adaptive sampling (add more training data where uncertain)
- Rigorous parameter inference (uncertainty propagation in MCMC)

## Example: N-body Cluster Evolution

- **Expensive**: N-body simulation (30 min/run, 3 parameters)
- **Training**: 100 LHS runs → 50 hours total
- **GP Training**: ~1 second
- **Predictions**: 10⁶ predictions in ~10 seconds
- **Speedup**: ~200,000× faster than direct simulation!

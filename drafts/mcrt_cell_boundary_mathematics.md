# Cell Boundary Crossing Mathematics for MCRT

## Ray-Grid Intersection in 3D Cartesian Coordinates

### Problem Statement

Given a photon packet at position $\vec{r}_0 = (x_0, y_0, z_0)$ traveling in direction $\hat{n} = (n_x, n_y, n_z)$ through a regular 3D grid, determine:
1. The distance to the next cell boundary
2. Which cell face will be crossed
3. The position where optical depth accumulation should continue

---

## Mathematical Formulation

### 1. Distance to Next Boundary in Each Dimension

For each spatial dimension, calculate the distance to the nearest boundary in the direction of travel:

#### X-dimension:
$$d_x = \begin{cases}
\dfrac{x_{\text{right}} - x_0}{n_x} & \text{if } n_x > 0 \quad \text{(moving toward +x)} \\[1em]
\dfrac{x_{\text{left}} - x_0}{n_x} & \text{if } n_x < 0 \quad \text{(moving toward -x)} \\[1em]
\infty & \text{if } n_x = 0 \quad \text{(parallel to x-planes)}
\end{cases}$$

#### Y-dimension:
$$d_y = \begin{cases}
\dfrac{y_{\text{top}} - y_0}{n_y} & \text{if } n_y > 0 \quad \text{(moving toward +y)} \\[1em]
\dfrac{y_{\text{bottom}} - y_0}{n_y} & \text{if } n_y < 0 \quad \text{(moving toward -y)} \\[1em]
\infty & \text{if } n_y = 0 \quad \text{(parallel to y-planes)}
\end{cases}$$

#### Z-dimension:
$$d_z = \begin{cases}
\dfrac{z_{\text{back}} - z_0}{n_z} & \text{if } n_z > 0 \quad \text{(moving toward +z)} \\[1em]
\dfrac{z_{\text{front}} - z_0}{n_z} & \text{if } n_z < 0 \quad \text{(moving toward -z)} \\[1em]
\infty & \text{if } n_z = 0 \quad \text{(parallel to z-planes)}
\end{cases}$$

---

## 2. Term Definitions

### Photon State Variables
- **$\vec{r}_0 = (x_0, y_0, z_0)$**: Current photon position [cm]
- **$\hat{n} = (n_x, n_y, n_z)$**: Unit propagation direction vector, where $|\hat{n}| = 1$

### Grid Parameters
- **$L$**: Total box size = $3.086 \times 10^{18}$ cm (1 parsec)
- **$N_{\text{grid}}$**: Number of cells per dimension = 128
- **$\Delta x = \Delta y = \Delta z$**: Cell size = $L/N_{\text{grid}}$ [cm]

### Current Cell Indices
For a position $(x_0, y_0, z_0)$, the cell indices are:
$$i_x = \left\lfloor \frac{x_0 + L/2}{\Delta x} \right\rfloor, \quad i_y = \left\lfloor \frac{y_0 + L/2}{\Delta y} \right\rfloor, \quad i_z = \left\lfloor \frac{z_0 + L/2}{\Delta z} \right\rfloor$$

### Cell Boundary Positions
For the current cell with indices $(i_x, i_y, i_z)$:

**X-boundaries:**
- $x_{\text{left}} = i_x \cdot \Delta x - L/2$ (left face)
- $x_{\text{right}} = (i_x + 1) \cdot \Delta x - L/2$ (right face)

**Y-boundaries:**
- $y_{\text{bottom}} = i_y \cdot \Delta y - L/2$ (bottom face)
- $y_{\text{top}} = (i_y + 1) \cdot \Delta y - L/2$ (top face)

**Z-boundaries:**
- $z_{\text{front}} = i_z \cdot \Delta z - L/2$ (front face)
- $z_{\text{back}} = (i_z + 1) \cdot \Delta z - L/2$ (back face)

---

## 3. Next Boundary Selection

The distance to the next cell boundary crossing is the minimum of the three calculated distances:

$$\boxed{d_{\text{next}} = \min(d_x, d_y, d_z)}$$

The crossed boundary type is determined by which distance equals $d_{\text{next}}$:

$$\text{Boundary crossed} = \begin{cases}
\text{x-face at } x_{\text{right/left}} & \text{if } d_{\text{next}} = d_x \\[0.5em]
\text{y-face at } y_{\text{top/bottom}} & \text{if } d_{\text{next}} = d_y \\[0.5em]
\text{z-face at } z_{\text{back/front}} & \text{if } d_{\text{next}} = d_z
\end{cases}$$

---

## 4. Optical Depth Accumulation

As the photon traverses each cell, accumulate optical depth:

$$\tau_{\text{accumulated}} = \tau_{\text{accumulated}} + \Delta\tau_{\text{cell}}$$

where the optical depth through the current cell (or portion thereof) is:

$$\Delta\tau_{\text{cell}} = \kappa_{\text{band}} \cdot \rho_{\text{dust}} \cdot d_{\text{step}}$$

### Terms:
- **$\kappa_{\text{band}}$**: Band-averaged mass absorption coefficient [cm²/g dust]
- **$\rho_{\text{dust}}$**: Dust density in current cell [g/cm³]
- **$d_{\text{step}}$**: Distance traveled in cell [cm]

---

## 5. Interaction Position Determination

The photon interacts when $\tau_{\text{accumulated}} \geq \tau_{\text{target}}$.

### Case 1: Interaction Within Current Cell
If $\tau_{\text{accumulated}} + \Delta\tau_{\text{cell}} \geq \tau_{\text{target}}$:

The interaction occurs at fractional distance through the cell:
$$f = \frac{\tau_{\text{target}} - \tau_{\text{accumulated}}}{\kappa_{\text{band}} \cdot \rho_{\text{dust}} \cdot d_{\text{next}}}$$

where $0 \leq f < 1$.

The interaction position is:
$$\vec{r}_{\text{interaction}} = \vec{r}_0 + f \cdot d_{\text{next}} \cdot \hat{n}$$

### Case 2: Continue to Next Cell
If $\tau_{\text{accumulated}} + \Delta\tau_{\text{cell}} < \tau_{\text{target}}$:

Update position to the cell boundary:
$$\vec{r}_{\text{new}} = \vec{r}_0 + (d_{\text{next}} + \epsilon) \cdot \hat{n}$$

where $\epsilon \sim 10^{-10}$ cm ensures numerical stability at boundaries.

---

## 6. Physical Interpretation

### Why Minimum Distance?
The photon travels in a straight line and encounters whichever boundary is geometrically closest along its path. The other boundaries are not reached during this step.

### Sign Convention
When a direction component is negative (e.g., $n_x < 0$), both the numerator and denominator in the distance calculation are negative, yielding a positive distance. This ensures all distances represent forward propagation along the ray.

### Parallel Motion
When $n_i = 0$ for any dimension $i$, the photon travels parallel to that set of boundaries and never crosses them, hence $d_i = \infty$.

### Numerical Precision
The epsilon adjustment ($\sim 10^{-10}$ cm) after boundary crossing prevents photons from becoming numerically "stuck" at cell interfaces due to floating-point precision limits.

---

## 7. Validation Tests

### Test 1: Axis-Aligned Ray
For $\hat{n} = (1, 0, 0)$:
- $d_x = \Delta x$ (regular spacing)
- $d_y = d_z = \infty$ (never crossed)

### Test 2: Diagonal Ray
For $\hat{n} = \frac{1}{\sqrt{3}}(1, 1, 1)$:
- Distances depend on position within starting cell
- Pattern of boundary crossings alternates between x, y, and z faces

### Test 3: Energy Conservation
$$\left|L_{\text{in}} - (L_{\text{absorbed}} + L_{\text{escaped}})\right| < 0.001 \cdot L_{\text{in}}$$

This must hold for all packet counts and grid resolutions.

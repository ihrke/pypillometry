# Foreshortening Correction for Pupil Size Measurements

## 1. Problem Statement

### 1.1 Physical Setup

**Eye-tracking measurement scenario:**
- **E** = (0,0,0): Eye position (origin)
- **C** = (c_x, c_y, c_z): Camera position (fixed, unknown direction, known distance r from eye)
- **S**: Computer screen (plane at fixed distance d from eye along z-axis)
- **T(x,y)** = (x, y, d): Gaze position on screen in mm from screen center

### 1.2 Foreshortening Effect

The camera measures apparent pupil size A(x,y,t), which is foreshortened relative to true pupil size A_0(t) depending on viewing angle:

$$A(x,y,t) = A_0(t) \cdot \cos(\alpha(x,y))$$

where:
- $A_0(t)$: **True pupil diameter/area** (intrinsic, viewing-angle independent)
- $A(x,y,t)$: **Measured pupil size** (affected by foreshortening)
- $\alpha(x,y)$: Angle between eye→camera vector ($\vec{EC}$) and eye→gaze vector ($\vec{ET}$)

**Physical interpretation:** When gaze direction $\vec{ET}$ is not aligned with camera direction $\vec{EC}$, the pupil appears elliptical/smaller to the camera. Maximum apparent size occurs when gazing directly toward camera ($\alpha = 0$).

### 1.3 Geometric Relationship

$$\cos(\alpha(x,y)) = \frac{\vec{EC} \cdot \vec{ET}}{|\vec{EC}| \cdot |\vec{ET}|} = \frac{c_x x + c_y y + c_z d}{r\sqrt{x^2 + y^2 + d^2}}$$

**Known parameters:**
- $r = |\vec{EC}|$: Eye-to-camera distance (measurable, typically 500-700 mm)
- $d$: Eye-to-screen distance (measurable, typically 600-800 mm)

**Unknown parameters:**
- $(c_x, c_y, c_z)$: Camera position with constraint $c_x^2 + c_y^2 + c_z^2 = r^2$

### 1.4 Camera Position Parameterization

Since r is known, camera position C has 2 degrees of freedom (spherical coordinates):

$$\begin{aligned}
c_x &= r \sin(\theta) \cos(\varphi) \\
c_y &= r \sin(\theta) \sin(\varphi) \\
c_z &= r \cos(\theta)
\end{aligned}$$

where:
- $\theta \in [0, \pi]$: Polar angle (elevation relative to z-axis)
- $\varphi \in [0, 2\pi)$: Azimuthal angle (horizontal direction)

Then:
$$\cos(\alpha(x,y; \theta, \varphi)) = \frac{\sin(\theta)\cos(\varphi) \cdot x + \sin(\theta)\sin(\varphi) \cdot y + \cos(\theta) \cdot d}{\sqrt{x^2 + y^2 + d^2}}$$

## 2. Experimental Protocol

### 2.1 Two-Phase Design

**Phase 1: Calibration** (Duration: ~2 minutes)
- Subject follows systematic gaze targets across screen
- Goal: Explore diverse gaze positions (x,y) to estimate camera geometry
- Spatial coverage: Full or majority of screen area

**Phase 2: Free Viewing / Task** (Duration: ~30 minutes)
- Subject performs natural task (reading, viewing, search, etc.)
- Gaze may be restricted to task-relevant screen regions
- Goal: Record pupil dynamics corrected for foreshortening

### 2.2 Data Structure

**Measurements:** $\mathcal{D} = \{(x_i, y_i, t_i, A_i)\}_{i=1}^N$

where for each sample i:
- $(x_i, y_i)$: Gaze coordinates on screen (mm from center)
- $t_i$: Timestamp
- $A_i$: Measured pupil size (pixels, mm, or arbitrary units)

**Partition:**
- $\mathcal{D}_{\text{cal}}$: Calibration phase $(t \leq T_{\text{cal}} \approx 2 \text{ min})$
- $\mathcal{D}_{\text{task}}$: Task phase $(t > T_{\text{cal}})$

## 3. Data Preprocessing

### 3.1 Quality Control

For raw pupil data sampled at 1000 Hz:

1. **Blink removal**: Exclude samples during blinks (pupil not visible)
2. **Artifact rejection**: Remove physically implausible values
3. **Gaze validation**: Ensure gaze coordinates are on screen

### 3.2 Temporal Filtering

Pupil dynamics of interest typically < 5 Hz (slow changes due to light, cognition, arousal):

1. **Low-pass filter**: Butterworth filter, cutoff $f_c = 3\text{-}5$ Hz
2. **Downsample**: 1000 Hz → 50 Hz (reduces computation 20×)

### 3.3 Valid Measurement Set

$$\mathcal{M} = \{(x_i, y_i, t_i, A_i) : \text{valid after preprocessing}\}$$

Partition: $\mathcal{M} = \mathcal{M}_{\text{cal}} \cup \mathcal{M}_{\text{task}}$

**Data completeness:**
$$\rho = \frac{|\mathcal{M}|}{N_{\text{total}}} \times 100\%$$

Typical: $\rho = 70\text{-}90\%$ (missing data due to blinks, track loss)

## 4. Temporal Model: True Pupil Size

### 4.1 B-Spline Basis Representation

Model smooth variation of true pupil size:

$$A_0(t; \mathbf{a}) = \sum_{k=1}^{K} a_k B_k(t)$$

where:
- $\{B_k(t)\}_{k=1}^K$: Cubic B-spline basis functions
- $\mathbf{a} = (a_1, \ldots, a_K)^T$: Basis coefficients (to be estimated)
- $K$: Number of basis functions (knots spaced 0.5-1 second apart)

**Knot placement:**
- For 2 min calibration: $K_{\text{cal}} \approx 120\text{-}240$ knots
- For 32 min total: $K_{\text{total}} \approx 1920\text{-}3840$ knots

### 4.2 Predicted Measurement Model

$$\hat{A}(x,y,t; \mathbf{a}, \theta, \varphi) = \left(\sum_{k=1}^K a_k B_k(t)\right) \cdot \cos(\alpha(x,y; \theta, \varphi))$$

**Parameter vector:**
$$\boldsymbol{\xi} = (\mathbf{a}, \theta, \varphi) = (a_1, \ldots, a_K, \theta, \varphi) \in \mathbb{R}^{K+2}$$

## 5. Two-Stage Calibration Algorithm

### 5.1 Stage 1: Initial Camera Geometry Estimation

**Objective:** Estimate camera position from calibration data with good spatial coverage.

$$\boldsymbol{\xi}_{\text{cal}}^* = \arg\min_{\boldsymbol{\xi}} \left[ L_{\text{data}}(\boldsymbol{\xi}; \mathcal{M}_{\text{cal}}) + \lambda_{\text{smooth}} R_{\text{smooth}}(\mathbf{a}) \right]$$

**Data fidelity term:**
$$L_{\text{data}}(\boldsymbol{\xi}; \mathcal{M}) = \sum_{(x_i,y_i,t_i,A_i) \in \mathcal{M}} \left[A_i - \hat{A}(x_i, y_i, t_i; \boldsymbol{\xi})\right]^2$$

**Smoothness regularization** (penalizes rapid pupil size changes):
$$R_{\text{smooth}}(\mathbf{a}) = \sum_{k=1}^{K-1} (a_{k+1} - a_k)^2$$

**Constraints:**
- $a_k > 0$ for all k (pupil size is positive)
- $0 \leq \theta \leq \pi$
- $0 \leq \varphi < 2\pi$

**Optimization:** Use L-BFGS-B with box constraints

**Output:** 
$$\boldsymbol{\xi}_{\text{cal}}^* = (\mathbf{a}_{\text{cal}}^*, \theta_{\text{cal}}^*, \varphi_{\text{cal}}^*)$$

### 5.2 Stage 2: Refinement Using Full Dataset

**Objective:** Improve estimates using all data while preventing camera geometry from drifting due to sparse spatial sampling in task phase.

$$\boldsymbol{\xi}_{\text{full}}^* = \arg\min_{\boldsymbol{\xi}} \Bigg[ w_{\text{cal}} L_{\text{data}}(\boldsymbol{\xi}; \mathcal{M}_{\text{cal}}) + w_{\text{task}} L_{\text{data}}(\boldsymbol{\xi}; \mathcal{M}_{\text{task}}) + \lambda_{\text{smooth}} R_{\text{smooth}}(\mathbf{a}) + \lambda_{\text{prior}} R_{\text{prior}}(\theta, \varphi) \Bigg]$$

**Weighted data terms:**
- $w_{\text{cal}} = 2\text{-}5$: Higher weight on calibration (diverse spatial sampling)
- $w_{\text{task}} = 1$: Standard weight on task data

**Geometric prior** (anchor camera position to Stage 1 estimate):
$$R_{\text{prior}}(\theta, \varphi) = (\theta - \theta_{\text{cal}}^*)^2 + (\varphi - \varphi_{\text{cal}}^*)^2$$

**Prior weight:** 
$$\lambda_{\text{prior}} \approx 0.1 \times |\mathcal{M}_{\text{cal}}|$$

**Temporal basis:** Use $K_{\text{total}}$ basis functions spanning entire session

**Output:**
$$\boldsymbol{\xi}_{\text{full}}^* = (\mathbf{a}_{\text{full}}^*, \theta_{\text{full}}^*, \varphi_{\text{full}}^*)$$

### 5.3 Final Camera Calibration

Store estimated camera geometry:

$$\begin{aligned}
\theta_{\text{camera}} &= \theta_{\text{full}}^* \\
\varphi_{\text{camera}} &= \varphi_{\text{full}}^* \\
\mathbf{C}_{\text{camera}} &= r \cdot (\sin\theta_{\text{camera}}\cos\varphi_{\text{camera}}, \, \sin\theta_{\text{camera}}\sin\varphi_{\text{camera}}, \, \cos\theta_{\text{camera}})
\end{aligned}$$

## 6. Foreshortening Correction

### 6.1 Correction Formula

For any measured pupil size $A(x,y,t)$ at gaze position (x,y):

$$A_0(t) = \frac{A(x,y,t)}{\cos(\alpha(x,y; \theta_{\text{camera}}, \varphi_{\text{camera}}))}$$

**Interpretation:** Dividing by $\cos(\alpha)$ "un-does" the foreshortening to recover true pupil size.

### 6.2 Practical Implementation

```python
def compute_foreshortening_factor(x_gaze, y_gaze, theta_cam, phi_cam, r, d):
    """
    Compute cos(alpha) for given gaze position and camera geometry.
    
    Parameters
    ----------
    x_gaze, y_gaze : float
        Gaze coordinates on screen (mm from center)
    theta_cam, phi_cam : float
        Camera position in spherical coordinates
    r : float
        Eye-to-camera distance (mm)
    d : float
        Eye-to-screen distance (mm)
    
    Returns
    -------
    cos_alpha : float
        Foreshortening factor (1.0 = no foreshortening)
    """
    # Camera position
    cx = r * np.sin(theta_cam) * np.cos(phi_cam)
    cy = r * np.sin(theta_cam) * np.sin(phi_cam)
    cz = r * np.cos(theta_cam)
    
    # Dot product of eye→camera and eye→gaze vectors
    numerator = cx * x_gaze + cy * y_gaze + cz * d
    
    # Product of vector magnitudes
    denominator = r * np.sqrt(x_gaze**2 + y_gaze**2 + d**2)
    
    return numerator / denominator


def correct_pupil_size(A_measured, x_gaze, y_gaze, calibration, threshold=0.15):
    """
    Correct measured pupil size for foreshortening.
    
    Parameters
    ----------
    A_measured : float or array
        Measured pupil size
    x_gaze, y_gaze : float or array
        Gaze coordinates (mm)
    calibration : dict
        Contains 'theta', 'phi', 'r', 'd' from calibration
    threshold : float
        Minimum cos(alpha) for reliable correction (default 0.15)
    
    Returns
    -------
    A0_corrected : float or array
        True pupil size (corrected for foreshortening)
    """
    cos_alpha = compute_foreshortening_factor(
        x_gaze, y_gaze,
        calibration['theta'], calibration['phi'],
        calibration['r'], calibration['d']
    )
    
    # Only correct when angle is not too oblique
    A0 = np.where(cos_alpha > threshold, 
                  A_measured / cos_alpha,
                  np.nan)
    
    return A0
```

### 6.3 Quality Control

**Reliability criterion:** Only apply correction when viewing angle is reasonable:

$$A_0^{\text{corrected}}(x,y,t) = 
\begin{cases}
\dfrac{A(x,y,t)}{\cos(\alpha(x,y))} & \text{if } \cos(\alpha(x,y)) > \epsilon \\[1em]
\text{NaN} & \text{otherwise}
\end{cases}$$

**Threshold selection:** $\epsilon = 0.15$ corresponds to $\alpha \approx 81°$

**Rationale:** 
- Small $\cos(\alpha)$ means extreme oblique viewing angle
- Correction factor $1/\cos(\alpha)$ becomes very large (e.g., > 6.7 for $\epsilon = 0.15$)
- Amplifies measurement noise substantially
- May indicate physiological limit where pupil is barely visible to camera

## 7. Validation

### 7.1 Model Fit Quality

**Residual analysis:**
$$e_i = A_i - \hat{A}(x_i, y_i, t_i; \boldsymbol{\xi}^*)$$

**Metrics:**
- Root mean square error: $\text{RMSE} = \sqrt{\frac{1}{|\mathcal{M}|}\sum_{i} e_i^2}$
- Coefficient of determination: $R^2 = 1 - \frac{\sum_i e_i^2}{\sum_i (A_i - \bar{A})^2}$
- Mean bias: $\bar{e} = \frac{1}{|\mathcal{M}|}\sum_i e_i$ (should be ≈ 0)

**Typical values:** $R^2 > 0.90$ indicates good fit

### 7.2 Spatial Consistency Check

**Test:** Corrected pupil size should be independent of gaze position.

For overlapping time periods with different gaze positions:
$$\text{Var}_{x,y}\left[A_0^{\text{corrected}}(x,y,t)\right] \ll \text{Var}_t\left[A_0^{\text{corrected}}(t)\right]$$

**Quantitative check:** Compute intra-class correlation:
$$\text{ICC} = \frac{\sigma_{\text{temporal}}^2}{\sigma_{\text{temporal}}^2 + \sigma_{\text{spatial}}^2}$$

Good correction: ICC > 0.95 (temporal variance dominates)

### 7.3 Camera Geometry Stability

**Compare Stage 1 and Stage 2 estimates:**
$$\Delta\theta = |\theta_{\text{full}}^* - \theta_{\text{cal}}^*|, \quad \Delta\varphi = |\varphi_{\text{full}}^* - \varphi_{\text{cal}}^*|$$

**Interpretation:**
- $\Delta\theta, \Delta\varphi < 5°$: Excellent stability
- $5° - 10°$: Acceptable (minor head movement or systematic drift)
- $> 10°$: Concerning (check for head motion, setup changes, or model violation)

### 7.4 Visual Inspection

Plot corrected vs. uncorrected pupil traces:

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

# Uncorrected
axes[0].plot(time, A_measured, alpha=0.5, label='Measured (uncorrected)')
axes[0].set_ylabel('Measured pupil size')
axes[0].legend()

# Corrected
axes[1].plot(time, A0_corrected, color='C1', label='Corrected')
axes[1].set_ylabel('True pupil size A₀(t)')
axes[1].set_xlabel('Time (s)')
axes[1].legend()

plt.tight_layout()
```

**Expected:** 
- Corrected trace should be smoother
- Gaze-dependent artifacts should be removed
- Physiological dynamics preserved

## 8. Typical Parameter Values

### 8.1 Geometric Setup
- Eye-to-screen distance: $d = 600\text{-}800$ mm
- Eye-to-camera distance: $r = 500\text{-}700$ mm (depends on eye-tracker model)
- Screen dimensions: $\pm 300\text{-}400$ mm horizontally, $\pm 200\text{-}300$ mm vertically

### 8.2 Camera Position
- Typical remote eye-trackers: Camera below screen center
  - $\theta \approx 90°\text{-}100°$ (slightly below horizontal)
  - $\varphi \approx 0°$ (centered horizontally)

### 8.3 Hyperparameters
- Smoothness weight: $\lambda_{\text{smooth}} = 10^{-2} \text{-} 10^{2}$ (tune via cross-validation)
- Prior weight: $\lambda_{\text{prior}} = 0.1 \times |\mathcal{M}_{\text{cal}}|$
- Calibration weight ratio: $w_{\text{cal}} / w_{\text{task}} = 2\text{-}5$
- Spline knots: 1-2 per second (adapt to signal bandwidth)

## 9. Complete Workflow

```
═══════════════════════════════════════════════════════════════
                    FORESHORTENING CORRECTION
                      FOR PUPIL SIZE MEASUREMENTS
═══════════════════════════════════════════════════════════════

INPUT:
  • Raw pupil data: {(x,y,t,A)} at 1000 Hz
  • Eye-to-camera distance: r (mm)
  • Eye-to-screen distance: d (mm)

STEP 1: PREPROCESSING
  ├─ Remove blinks and artifacts
  ├─ Low-pass filter (3-5 Hz)
  ├─ Downsample to 50 Hz
  └─ Create valid measurement set M

STEP 2: PARTITION DATA
  ├─ M_cal: Calibration phase (first ~2 min, diverse gaze)
  └─ M_task: Task phase (remaining ~30 min)

STEP 3: STAGE 1 CALIBRATION
  ├─ Fit model to M_cal only
  ├─ Estimate camera geometry: θ*_cal, φ*_cal
  └─ Validate fit quality (R² > 0.90)

STEP 4: STAGE 2 REFINEMENT
  ├─ Fit model to full dataset M = M_cal ∪ M_task
  ├─ Use Stage 1 geometry as prior
  ├─ Weight calibration data higher (w_cal = 2-5)
  └─ Output: θ*_full, φ*_full, full A₀(t)

STEP 5: APPLY CORRECTION
  For each measurement A(x,y,t):
    ├─ Compute: cos(α(x,y)) using θ*_full, φ*_full
    ├─ Correct: A₀(t) = A(x,y,t) / cos(α(x,y))
    └─ Flag unreliable: if cos(α) < 0.15, set to NaN

STEP 6: VALIDATION
  ├─ Check residuals (RMSE, R²)
  ├─ Verify spatial consistency (ICC > 0.95)
  ├─ Assess geometry stability (Δθ, Δφ < 10°)
  └─ Visual inspection of corrected traces

OUTPUT:
  • Corrected pupil size time series: A₀(t)
  • Camera calibration: (θ, φ, r, d)
  • Quality metrics and validation results

═══════════════════════════════════════════════════════════════
```

## 10. Extensions and Considerations

### 10.1 Head Movement

If subject moves head during session:
- Eye-camera distance r changes → violates fixed-geometry assumption
- **Solution 1:** Track head position and update r, d in real-time
- **Solution 2:** Use head-mounted eye-tracker (r, d constant in head reference frame)
- **Solution 3:** Model time-varying camera position (advanced)

### 10.2 Binocular Eye-Tracking with Nose-Centered Coordinates

For dual-eye systems, using both eyes simultaneously provides significant advantages for geometry estimation and correction reliability.

#### 10.2.1 Coordinate System

**Nose-centered reference frame:**
- Origin at nose midpoint: $N = (0,0,0)$
- Left eye: $E_L = (-\text{IPD}/2, 0, 0)$
- Right eye: $E_R = (+\text{IPD}/2, 0, 0)$
- Inter-pupillary distance (IPD): typically 60-65 mm (measurable)
- Camera: $C = (c_x, c_y, c_z)$ with $|\vec{NC}| = r$
- Screen: plane at distance $d$ from nose (assumes frontal viewing)

**Key advantage:** Both eyes constrain the same camera position $C$, providing stereo-like geometric information.

#### 10.2.2 Modified Geometric Model

For eye $i \in \{L, R\}$ viewing screen position $(x,y)$:

$$\vec{E_i} = \begin{cases}
(-\text{IPD}/2, 0, 0) & \text{if } i = L \\
(+\text{IPD}/2, 0, 0) & \text{if } i = R
\end{cases}$$

$$\vec{E_iC} = \vec{C} - \vec{E_i} = (c_x - e_{i,x}, c_y, c_z)$$

$$\vec{E_iT} = (x, y, d) - \vec{E_i} = (x - e_{i,x}, y, d)$$

**Foreshortening factor:**
$$\cos(\alpha_i(x,y)) = \frac{\vec{E_iC} \cdot \vec{E_iT}}{|\vec{E_iC}| \cdot |\vec{E_iT}|}$$

$$= \frac{(c_x - e_{i,x})(x - e_{i,x}) + c_y y + c_z d}{\sqrt{(c_x - e_{i,x})^2 + c_y^2 + c_z^2} \cdot \sqrt{(x - e_{i,x})^2 + y^2 + d^2}}$$

**Camera parameterization** (still 2 DOF):
$$\begin{aligned}
c_x &= r \sin(\theta) \cos(\varphi) \\
c_y &= r \sin(\theta) \sin(\varphi) \\
c_z &= r \cos(\theta)
\end{aligned}$$

where $r$ is now the nose-to-camera distance.

#### 10.2.3 Pupil Dynamics Model

Both pupils respond identically to cognitive/arousal state (physiologically coupled):

$$\begin{aligned}
A_L(x,y,t) &= A_0(t) \cdot \cos(\alpha_L(x,y; \theta, \varphi)) \\
A_R(x,y,t) &= A_0(t) \cdot \cos(\alpha_R(x,y; \theta, \varphi))
\end{aligned}$$

- Single shared spline: $A_0(t) = \sum_{k=1}^K a_k B_k(t)$
- Parameter count: $K + 2$ (same as monocular)
- Physiologically plausible for normal binocular vision

#### 10.2.4 Modified Optimization Problem

**Objective function:**

$$\min_{\mathbf{a}, \theta, \varphi} \left[ \sum_{i \in \{L,R\}} L_{\text{data}}(\mathbf{a}, \theta, \varphi; \mathcal{M}_i) + \lambda_{\text{smooth}} R_{\text{smooth}}(\mathbf{a}) \right]$$

where:
$$L_{\text{data}}(\mathbf{a}, \theta, \varphi; \mathcal{M}_i) = \sum_{(x,y,t,A) \in \mathcal{M}_i} \left[A - \left(\sum_k a_k B_k(t)\right) \cos(\alpha_i(x,y; \theta, \varphi))\right]^2$$

**Measurement set:**
$$\mathcal{M} = \mathcal{M}_L \cup \mathcal{M}_R = \{(i, x_j, y_j, t_j, A_{i,j})\}$$

where $i \in \{L, R\}$ indicates eye.

**Two-stage approach:**
- Stage 1: Fit calibration data from both eyes → $(\theta^*_{\text{cal}}, \varphi^*_{\text{cal}})$
- Stage 2: Fit full dataset with geometric prior
- Both eyes constrain same $(\theta, \varphi)$ throughout

#### 10.2.5 Advantages of Binocular Approach

1. **More data, same parameters:**
   - ~2× measurements for same screen locations
   - Better statistical power
   - Expected: 20-40% reduction in parameter uncertainty

2. **Geometric cross-validation:**
   - Both eyes must agree on camera position
   - Natural consistency check
   - Reduces local minima in optimization

3. **Complementary viewing angles:**
   - For same screen point, left/right eyes have different $\alpha$
   - Provides stereo-like constraints on $C$
   - Especially valuable when task-phase gaze is spatially restricted

4. **Robustness to missing data:**
   - If one eye has blinks/artifacts, other eye still provides information
   - Better temporal coverage

#### 10.2.6 Implementation Notes

**Known parameters:**
- IPD: Measure from calibration data or use typical value (62-64 mm)
- $d$: Eye-to-screen distance (assumes frontal viewing, equal for both eyes)
- $r$: Nose-to-camera distance (eye-tracker specification or measured)

**Assumptions:**
- Head is frontal (nose perpendicular to screen)
- Both eyes equidistant from screen
- Single camera (or symmetric dual-camera setup)
- If head is tilted, IPD vector rotates → requires head pose tracking

**Validation:**
- Check $A_{0,L}^{\text{corrected}}(t) \approx A_{0,R}^{\text{corrected}}(t)$ (should be highly correlated)
- Strong disagreement may indicate anisocoria or model violation

**Correction application:**
Each eye corrected using the same geometric parameters:
$$A_{0,i}^{\text{corrected}}(t) = \frac{A_i(x,y,t)}{\cos(\alpha_i(x,y; \theta^*, \varphi^*))}$$

**Final estimate:**
$$A_0^{\text{final}}(t) = \frac{1}{2}\left[A_{0,L}^{\text{corrected}}(t) + A_{0,R}^{\text{corrected}}(t)\right]$$

Averaging provides noise reduction and validates that both eyes yield consistent estimates.

#### 10.2.7 Comparison with Monocular Approach

| Aspect | Monocular | Binocular |
|--------|-----------|-----------|
| Coordinate origin | Eye | Nose |
| Parameters | $K + 2$ | $K + 2$ |
| Data per timepoint | 1 measurement | 2 measurements |
| Geometry uncertainty | Baseline | ~30% lower |
| Computational cost | 1× | ~1.5× |
| Physiological validity | N/A | High (coupled pupils) |

**Recommendation:** Use the binocular approach when both eyes are tracked. It provides better geometric constraints with minimal additional computational cost and maintains physiological plausibility by assuming coupled pupil responses.

#### 10.2.8 Example: Left Eye Correction Factor

For left eye at gaze position $(x,y)$ with IPD = 63 mm:

$$e_{L,x} = -31.5 \text{ mm}$$

$$\cos(\alpha_L(x,y)) = \frac{[r\sin\theta\cos\varphi + 31.5](x + 31.5) + r\sin\theta\sin\varphi \cdot y + r\cos\theta \cdot d}{\sqrt{[r\sin\theta\cos\varphi + 31.5]^2 + [r\sin\theta\sin\varphi]^2 + [r\cos\theta]^2} \cdot \sqrt{(x+31.5)^2 + y^2 + d^2}}$$

Right eye analogous with $e_{R,x} = +31.5$ mm.

### 10.3 Alternative Pupil Metrics

Algorithm generalizes to:
- **Pupil diameter** (most common)
- **Pupil area** (more robust to ellipse fitting)
- **Pupil aspect ratio** (additional correction may be needed)

### 10.4 Multi-Session Stability

For longitudinal studies:
- **Recalibrate** if setup changes (camera moved, new subject session)
- **Reuse calibration** if setup is mechanically stable
- **Track geometry drift** across days/weeks

### 10.5 Computational Efficiency

For very long recordings (hours):
- Use sparse temporal basis (fewer knots)
- Consider piecewise fitting (segment into chunks)
- Parallelize across segments

### 10.6 Incorporating Gaze Calibration Uncertainty

If eye-tracker calibration quality estimates are available (i.e., spatial uncertainty in gaze position measurements), this information can significantly improve the robustness and statistical optimality of the foreshortening correction.

#### 10.6.1 Weighted Least Squares Formulation

**Standard objective function:**
$$L_{\text{data}} = \sum_i \left[A_i - \hat{A}(x_i, y_i, t_i)\right]^2$$

**Weighted objective function:**
$$L_{\text{weighted}} = \sum_i w_i \left[A_i - \hat{A}(x_i, y_i, t_i)\right]^2$$

where $w_i$ accounts for measurement uncertainty at sample $i$.

#### 10.6.2 Uncertainty Propagation

Gaze position uncertainty propagates to the foreshortening factor through the chain rule:

$$\sigma_{\cos\alpha}^2 \approx \left(\frac{\partial \cos\alpha}{\partial x}\right)^2 \sigma_x^2 + \left(\frac{\partial \cos\alpha}{\partial y}\right)^2 \sigma_y^2$$

where $\sigma_x$ and $\sigma_y$ are gaze position uncertainties (typically provided by eye-tracker calibration).

**Key insight:** Gaze measurements with higher uncertainty contribute less reliable information about the foreshortening factor and should receive lower weight in the optimization.

#### 10.6.3 Optimal Weighting

For sample $i$ with gaze uncertainty $(\sigma_{x,i}, \sigma_{y,i})$:

$$w_i = \frac{1}{\sigma_{\text{total},i}^2}$$

where the total measurement variance combines pupil and gaze-induced uncertainty:

$$\sigma_{\text{total},i}^2 = \sigma_{A}^2 + \left[\frac{\partial \hat{A}}{\partial \cos\alpha}\right]^2 \sigma_{\cos\alpha,i}^2$$

**Simplified approximation** (assuming pupil measurement noise dominates or is constant):

$$w_i \propto \frac{1}{\sigma_{\cos\alpha,i}^2} \propto \frac{1}{\left(\frac{\partial \cos\alpha}{\partial x}\right)^2 \sigma_{x,i}^2 + \left(\frac{\partial \cos\alpha}{\partial y}\right)^2 \sigma_{y,i}^2}$$

#### 10.6.4 Practical Implementation

**Modified optimization problem:**

$$\min_{\boldsymbol{\xi}} \left[ \sum_i w_i \left[A_i - \hat{A}(x_i, y_i, t_i; \boldsymbol{\xi})\right]^2 + \lambda_{\text{smooth}} R_{\text{smooth}}(\mathbf{a}) + \lambda_{\text{prior}} R_{\text{prior}}(\theta, \varphi) \right]$$

**Weight computation algorithm:**

1. For each measurement, obtain gaze uncertainty $(\sigma_x, \sigma_y)$ from eye-tracker calibration
2. Compute gradients of $\cos(\alpha(x,y))$ with respect to gaze position
3. Propagate uncertainty to $\sigma_{\cos\alpha}$
4. Set weight $w = 1/\sigma_{\cos\alpha}^2$ (normalized across all samples)

**Thresholding:** Optionally exclude samples where gaze uncertainty exceeds a threshold (e.g., $\sigma_{\text{gaze}} > 2°$ visual angle or 50-100 pixels), indicating very poor tracking quality.

#### 10.6.5 Benefits

**1. Better geometry estimation:**
- Downweight samples with poor gaze tracking
- Prevents miscalibrated gaze points from biasing camera position estimates
- Especially important during calibration phase where spatial diversity is critical

**2. Robustness to outliers:**
- Automatically reduces influence of poorly tracked samples
- Natural quality control without hard thresholds

**3. Spatially-varying quality:**
- If gaze tracking is worse at screen periphery (common), central fixations naturally receive higher weight
- Optimal use of available information

**4. Eye-specific weighting (binocular):**
- If left/right eyes have different calibration quality, weight appropriately
- Still estimate shared $A_0(t)$ but with statistically optimal weighting

**5. Temporal degradation:**
- If gaze quality degrades over long recordings, later data automatically downweighted
- More robust parameter estimates

#### 10.6.6 When This Extension Helps Most

**High impact scenarios:**
✓ Calibration quality varies spatially (worse at screen edges)  
✓ Some calibration trials failed or were marginal  
✓ One eye tracks better than the other (binocular case)  
✓ Long recordings where tracking quality degrades  
✓ Subjects with poor eye-tracking (e.g., elderly, clinical populations)

**Lower impact scenarios:**
- Uniform high-quality gaze tracking across all samples
- Gaze uncertainty << variation in foreshortening effect across viewing angles

#### 10.6.7 Validation

**Check improvement:**
- Compare weighted vs. unweighted fit quality (R², residuals)
- Plot residuals vs. gaze uncertainty (should see no correlation after weighting)
- Check if parameter uncertainty estimates decrease

**Expected improvement:**
- 10-30% reduction in parameter uncertainty if gaze errors are spatially heterogeneous
- Improved fit quality (R²) by 2-5% typically
- More stable geometry estimates across multiple fits

#### 10.6.8 Implementation Notes

**Obtaining gaze uncertainty:**
- Most eye-trackers provide calibration/validation accuracy metrics
- Can be global (single value) or spatially-varying (map across screen)
- Typical values: 0.5° (excellent) to 2° (poor) visual angle

**Gradient computation:**
For efficient implementation, compute gradients analytically:

$$\frac{\partial \cos\alpha}{\partial x} = \frac{c_x}{r \cdot |\vec{ET}|} - \frac{\cos\alpha \cdot (x - e_x)}{|\vec{ET}|^2}$$

$$\frac{\partial \cos\alpha}{\partial y} = \frac{c_y}{r \cdot |\vec{ET}|} - \frac{\cos\alpha \cdot y}{|\vec{ET}|^2}$$

where $|\vec{ET}| = \sqrt{(x - e_x)^2 + y^2 + d^2}$ and $e_x$ is the eye x-position (0 for monocular, ±IPD/2 for binocular).

**Computational cost:**
- Minimal overhead (~5-10% increase in optimization time)
- Gradient computations are vectorizable

**Alternative: Robust regression:**
If explicit uncertainty estimates are unavailable, consider robust M-estimators (e.g., Huber loss) that automatically downweight outliers.

### 10.7 Extended Geometry: Screen Misalignment and Eye Offset

Real experimental setups involve unavoidable geometric imperfections: the eye is rarely perfectly centered with the screen midpoint, and the screen plane is typically not perfectly perpendicular to the primary viewing direction. This section extends the algorithm to estimate these alignment parameters from calibration data.

#### 10.7.1 Problem Motivation

**Idealized assumptions** (Sections 1-9):
- Eye positioned at screen center: $\vec{ET}(0,0) = (0, 0, d)$
- Screen perpendicular to z-axis: points $(x, y)$ map to 3D positions $(x, y, d)$

**Realistic setup imperfections**:
- Eye offset from screen center by $(\Delta x, \Delta y)$ (typically 0-100 mm)
- Screen tilted by small angles relative to eye-perpendicular orientation
  - Pitch $\alpha$: tilt about horizontal axis (screen leans forward/back)
  - Yaw $\beta$: tilt about vertical axis (screen rotates left/right)
  - Typical: $|\alpha|, |\beta| < 10°$

**Consequences if ignored**:
- Systematic bias in corrected pupil size
- Residual gaze-dependent artifacts
- Poorer model fit quality

**Solution**: Jointly estimate 4 additional geometric parameters $(\Delta x, \Delta y, \alpha, \beta)$ from calibration data with diverse spatial coverage.

**Note on roll angle**: A screen has 3 rotational degrees of freedom (pitch, yaw, roll), but roll (rotation about the depth axis) has negligible effect on foreshortening and is omitted for parsimony.

#### 10.7.2 Coordinate System and Reference Frame

**Eye-centered reference frame** (unchanged):
- Origin: **E** = (0, 0, 0)
- Z-axis: Primary viewing direction (perpendicular to screen in idealized case)
- X-axis: Rightward (subject's perspective)
- Y-axis: Upward

**Screen coordinate system**:
- Measured gaze coordinates $(x, y)$ are in the screen's intrinsic 2D coordinate system
- Origin at screen center (by convention)
- Units: millimeters

**Screen pose in eye frame**: Defined by:
1. **Translation**: Screen center position $\vec{S}_0 = (\Delta x, \Delta y, d)$
2. **Orientation**: Euler angles $(\alpha, \beta)$ relative to frontal (eye-perpendicular)

#### 10.7.2.1 Practical Measurement of Eye-Screen Distance

**Challenge**: If eye offset $(\Delta x, \Delta y)$ is unknown, measuring the "eye-to-screen distance" is ambiguous.

**Recommended approach: Perpendicular distance**

Define $d$ as the **perpendicular distance** from eye to screen plane:
- Measure with a ruler held perpendicular to the screen surface
- Independent of horizontal/vertical eye position
- Geometrically: $d$ is the projection of the eye position onto the screen normal vector

**Alternative: Fixed reference point measurement**

If perpendicular measurement is impractical, measure eye distance to a **fixed screen reference point** and adjust the model:

**Option A: Eye to screen center**

Measure: $d_{\text{measured}} = |\vec{ES}_0|$ (eye to physical screen center)

Relationship:
$$d_{\text{measured}}^2 = \Delta x^2 + \Delta y^2 + d^2$$

In optimization, **fix** $d_{\text{measured}}$ and treat $d$ as an additional parameter:
$$d = \sqrt{d_{\text{measured}}^2 - \Delta x^2 - \Delta y^2}$$

Constraint: $\Delta x^2 + \Delta y^2 < d_{\text{measured}}^2$ (eye cannot be beyond screen distance)

**Option B: Eye to screen corner** (e.g., upper-left)

Measure: $d_{\text{corner}} = |\vec{E} - \vec{P}_{\text{corner}}|$ 

For upper-left corner at screen coordinates $(-w/2, h/2)$ where $w, h$ are screen width/height:

$$d_{\text{corner}}^2 = (\Delta x - w/2)^2 + (\Delta y + h/2)^2 + d^2$$

Solve for $d$:
$$d = \sqrt{d_{\text{corner}}^2 - (\Delta x - w/2)^2 - (\Delta y + h/2)^2}$$

**Option C: Estimate $d$ from calibration data**

Treat $d$ as an **additional free parameter** (not pre-measured):
- Parameter vector: $\boldsymbol{\xi}_{\text{geo}} = (\theta, \varphi, \Delta x, \Delta y, \alpha, \beta, d)$ (7 parameters)
- Requires strong spatial diversity in calibration
- Prior: $d \sim \mathcal{N}(650, 100^2)$ mm (typical viewing distance ± uncertainty)
- Validation: Compare estimated $d$ to physical measurement

**Recommendation by setup type**:

| Setup | Recommended approach | Notes |
|-------|---------------------|-------|
| Chin rest (fixed head) | Perpendicular distance | Most accurate; easy to measure with ruler |
| Remote eye-tracker (free head) | Estimate from calibration | $d$ varies with head position anyway |
| Fixed monitor | Screen center distance | Easy to measure; apply Option A |
| Laptop/tablet | Corner distance + estimate | Screen size known; Option B or C |

**Implementation note**: For small offsets ($|\Delta x|, |\Delta y| < 50$ mm) and typical viewing distances ($d \approx 600\text{-}700$ mm), the difference between perpendicular distance and center distance is small ($< 5$ mm). For most applications, measuring approximate eye-to-screen-center distance and treating it as $d$ introduces negligible error.

#### 10.7.3 Mathematical Model: Screen Tilt Transformation

For a gaze point at screen coordinates $(x_s, y_s)$ (screen frame), compute 3D position in eye frame:

**Step 1: Homogeneous coordinates in screen frame**

$$\vec{P}_{\text{screen}} = \begin{pmatrix} x_s \\ y_s \\ 0 \end{pmatrix}$$

**Step 2: Rotation matrices** (applied in order: pitch then yaw)

Pitch rotation about x-axis:

$$\mathbf{R}_{\text{pitch}}(\alpha) = \begin{pmatrix}
1 & 0 & 0 \\
0 & \cos\alpha & -\sin\alpha \\
0 & \sin\alpha & \cos\alpha
\end{pmatrix}$$

Yaw rotation about y-axis:

$$\mathbf{R}_{\text{yaw}}(\beta) = \begin{pmatrix}
\cos\beta & 0 & \sin\beta \\
0 & 1 & 0 \\
-\sin\beta & 0 & \cos\beta
\end{pmatrix}$$

Combined rotation:

$$\mathbf{R}(\alpha, \beta) = \mathbf{R}_{\text{yaw}}(\beta) \cdot \mathbf{R}_{\text{pitch}}(\alpha)$$

**Step 3: Transform to eye frame**

Screen normal in eye frame (initially $\hat{z} = (0,0,1)$ for perpendicular screen):

$$\hat{n}_{\text{screen}} = \mathbf{R}(\alpha, \beta) \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix}$$

Point on screen in eye frame:

$$\vec{T}(x_s, y_s) = \mathbf{R}(\alpha, \beta) \begin{pmatrix} x_s \\ y_s \\ 0 \end{pmatrix} + \begin{pmatrix} \Delta x \\ \Delta y \\ d \end{pmatrix}$$

**Expanded form**:

$$\begin{aligned}
T_x(x_s, y_s) &= x_s \cos\beta + y_s \sin\alpha \sin\beta + \Delta x \\
T_y(x_s, y_s) &= y_s \cos\alpha - x_s \sin\alpha \sin\beta + \Delta y \\
T_z(x_s, y_s) &= -x_s \sin\beta + y_s \sin\alpha \cos\beta + d \cos\alpha \cos\beta
\end{aligned}$$

**Small-angle approximation** (valid for $|\alpha|, |\beta| < 0.2$ rad $\approx 11°$):

$$\cos\alpha \approx 1 - \frac{\alpha^2}{2}, \quad \sin\alpha \approx \alpha$$

$$\cos\beta \approx 1 - \frac{\beta^2}{2}, \quad \sin\beta \approx \beta$$

Linearized (first-order):

$$\begin{aligned}
T_x &\approx x_s + y_s \alpha \beta + \Delta x \\
T_y &\approx y_s + \Delta y \\
T_z &\approx d - x_s \beta + y_s \alpha
\end{aligned}$$

#### 10.7.4 Modified Foreshortening Model

**Gaze vector** (eye to screen point):

$$\vec{ET}(x_s, y_s) = \vec{T}(x_s, y_s) - \vec{E} = \vec{T}(x_s, y_s)$$

**Camera vector** (eye to camera, unchanged):

$$\vec{EC} = (c_x, c_y, c_z) = r(\sin\theta\cos\varphi, \sin\theta\sin\varphi, \cos\theta)$$

**Foreshortening factor**:

$$\cos(\alpha_{\text{view}}(x_s, y_s; \boldsymbol{\xi}_{\text{geo}})) = \frac{\vec{EC} \cdot \vec{ET}(x_s, y_s)}{|\vec{EC}| \cdot |\vec{ET}(x_s, y_s)|}$$

where geometric parameter vector:

$$\boldsymbol{\xi}_{\text{geo}} = (\theta, \varphi, \Delta x, \Delta y, \alpha, \beta)$$

**Note**: $\alpha$ here is the screen pitch angle; $\alpha_{\text{view}}$ is the viewing angle (foreshortening).

**Full predicted measurement**:

$$\hat{A}(x_s, y_s, t; \mathbf{a}, \boldsymbol{\xi}_{\text{geo}}) = \left(\sum_{k=1}^K a_k B_k(t)\right) \cdot \cos(\alpha_{\text{view}}(x_s, y_s; \boldsymbol{\xi}_{\text{geo}}))$$

#### 10.7.5 Fixed Camera-Screen Rig (Recommended Parameterization)

**Physical setup**: Camera rigidly mounted to screen (common for remote eye-trackers).

**Implication**: Camera position relative to screen is constant in the **screen reference frame**.

**Parameterization**:
- Camera position in screen frame: $\vec{C}_{\text{rel}} = (c_x^{\text{rel}}, c_y^{\text{rel}}, c_z^{\text{rel}})$
- Typically: camera below/above screen center
  - Example: $c_x^{\text{rel}} = 0$ (centered horizontally)
  - $c_y^{\text{rel}} = -300$ mm (below screen)
  - $c_z^{\text{rel}} = 0$ (flush with screen)
- Constraint: $|\vec{C}_{\text{rel}}| = r_{\text{rel}}$ (known from eye-tracker specs)

**Camera position in eye frame**:

$$\vec{C}_{\text{eye}} = \mathbf{R}(\alpha, \beta) \vec{C}_{\text{rel}} + \begin{pmatrix} \Delta x \\ \Delta y \\ d \end{pmatrix}$$

**Advantage**: Camera orientation automatically consistent with screen orientation. Reduces parameter correlations.

**Parameter vector** (if using fixed rig parameterization):

$$\boldsymbol{\xi}_{\text{geo}} = (\theta_{\text{rel}}, \varphi_{\text{rel}}, \Delta x, \Delta y, \alpha, \beta)$$

where $(\theta_{\text{rel}}, \varphi_{\text{rel}})$ define camera position in screen-centered spherical coordinates.

#### 10.7.6 Prior Distributions for Soft Constraints

Instead of hard bounds, use **probabilistic priors** that encode expected parameter ranges while allowing data-driven deviations.

**Prior specification**:

**1. Eye offset from screen center**:

$$\Delta x \sim \mathcal{N}(0, \sigma_{\Delta x}^2), \quad \Delta y \sim \mathcal{N}(0, \sigma_{\Delta y}^2)$$

Typical: $\sigma_{\Delta x} = \sigma_{\Delta y} = 50$ mm (allows ±100 mm at 2σ)

**2. Screen tilt angles**:

If tilt direction unknown (symmetric):

$$\alpha \sim \mathcal{N}(0, \sigma_{\alpha}^2), \quad \beta \sim \mathcal{N}(0, \sigma_{\beta}^2)$$

If tilt direction known (e.g., screen typically leans back):

$$\alpha \sim \mathcal{N}(\mu_{\alpha}, \sigma_{\alpha}^2)$$

where $\mu_{\alpha} < 0$ (negative pitch = backward tilt)

For one-sided constraints (e.g., "screen cannot tilt forward"):

$$\alpha \sim -\text{HalfNormal}(\sigma_{\alpha}^2) \quad \text{(constrained to } \alpha \leq 0\text{)}$$

Typical: $\sigma_{\alpha} = \sigma_{\beta} = 5° = 0.087$ rad (allows ±10° at 2σ)

**3. Camera position** (if not using fixed rig):

Weakly informative priors on spherical coordinates:

$$\theta \sim \mathcal{N}(\mu_{\theta}, \sigma_{\theta}^2), \quad \varphi \sim \mathcal{N}(0, \sigma_{\varphi}^2)$$

Example: Camera known to be below screen
- $\mu_{\theta} = 100° = 1.75$ rad (slightly below horizontal)
- $\sigma_{\theta} = 20° = 0.35$ rad (loose constraint)

**Prior regularization term**:

$$R_{\text{prior}}(\boldsymbol{\xi}_{\text{geo}}) = \frac{(\Delta x - \mu_{\Delta x})^2}{2\sigma_{\Delta x}^2} + \frac{(\Delta y - \mu_{\Delta y})^2}{2\sigma_{\Delta y}^2} + \frac{(\alpha - \mu_{\alpha})^2}{2\sigma_{\alpha}^2} + \frac{(\beta - \mu_{\beta})^2}{2\sigma_{\beta}^2}$$

(Assumes Gaussian priors; for HalfNormal or other distributions, modify accordingly)

**Interpretation**: This is equivalent to adding a scaled negative log-prior to the loss function, implementing **maximum a posteriori (MAP)** estimation.

#### 10.7.7 Modified Optimization Problem

**Full parameter vector**:

$$\boldsymbol{\xi} = (\mathbf{a}, \boldsymbol{\xi}_{\text{geo}}) = (a_1, \ldots, a_K, \theta, \varphi, \Delta x, \Delta y, \alpha, \beta) \in \mathbb{R}^{K+6}$$

**Stage 1: Calibration phase estimation**

$$\boldsymbol{\xi}_{\text{cal}}^* = \arg\min_{\boldsymbol{\xi}} \Bigg[ L_{\text{data}}(\boldsymbol{\xi}; \mathcal{M}_{\text{cal}}) + \lambda_{\text{smooth}} R_{\text{smooth}}(\mathbf{a}) + \lambda_{\text{prior}} R_{\text{prior}}(\boldsymbol{\xi}_{\text{geo}}) \Bigg]$$

**Stage 2: Full dataset refinement**

$$\boldsymbol{\xi}_{\text{full}}^* = \arg\min_{\boldsymbol{\xi}} \Bigg[ w_{\text{cal}} L_{\text{data}}(\boldsymbol{\xi}; \mathcal{M}_{\text{cal}}) + w_{\text{task}} L_{\text{data}}(\boldsymbol{\xi}; \mathcal{M}_{\text{task}}) + \lambda_{\text{smooth}} R_{\text{smooth}}(\mathbf{a}) + \lambda_{\text{prior}}^{(2)} R_{\text{prior}}(\boldsymbol{\xi}_{\text{geo}}) \Bigg]$$

where:
- Stage 1 prior weight: $\lambda_{\text{prior}} = 1$ (normalized with data term)
- Stage 2 prior weight: $\lambda_{\text{prior}}^{(2)} = 0.1 \times |\mathcal{M}_{\text{cal}}|$ (stronger anchoring)

**Prior mean specification**:
- Stage 1: Use weakly informative priors (broad $\sigma$, uninformative $\mu$)
- Stage 2: Optionally center priors at Stage 1 estimates: $\mu_{\alpha} = \alpha_{\text{cal}}^*$, etc.

#### 10.7.8 Identifiability and Parameter Correlations

**Concern**: Are 6 geometric parameters uniquely identifiable, or do some trade off?

**Potential correlations**:
1. $\Delta x \leftrightarrow \varphi$: Horizontal eye offset vs. camera azimuth
2. $\Delta y \leftrightarrow \theta$: Vertical eye offset vs. camera elevation
3. $\alpha, \beta \leftrightarrow \theta, \varphi$: Screen tilt vs. camera position

**Mitigation strategies**:

1. **Spatial diversity in calibration**: 
   - Gaze positions spanning full screen break degeneracies
   - Different $(x_s, y_s)$ probe different aspects of geometry

2. **Temporal information**: 
   - Pupil dynamics $A_0(t)$ are independent of geometry
   - Separates temporal (pupil) from spatial (geometry) effects

3. **Prior regularization**: 
   - Soft constraints prevent extreme parameter values
   - Priors encode approximate knowledge of setup

4. **Fixed rig constraint** (if applicable):
   - Reduces degrees of freedom
   - Camera and screen orientations coupled

**Validation**: Check parameter uncertainty estimates from optimization:
- Compute Hessian at optimum: $\mathbf{H} = \nabla^2 L(\boldsymbol{\xi}^*)$
- Parameter covariance: $\text{Cov}(\boldsymbol{\xi}) \approx \mathbf{H}^{-1}$
- Standard errors: $\text{SE}(\xi_i) = \sqrt{[\mathbf{H}^{-1}]_{ii}}$

**Rule of thumb**: If $\text{SE}(\xi_i) < 0.5 |\xi_i^*|$, parameter is well-identified.

#### 10.7.9 Interpretation and Physical Validation

**Output of extended calibration**:

$$\boldsymbol{\xi}_{\text{geo}}^* = (\theta^*, \varphi^*, \Delta x^*, \Delta y^*, \alpha^*, \beta^*)$$

**Interpretation**:
- $\Delta x^* = -35$ mm: "Eye is 35 mm left of screen center"
- $\Delta y^* = +12$ mm: "Eye is 12 mm above screen center"
- $\alpha^* = -0.09$ rad $= -5.2°$: "Screen tilts backward (top away from eye)"
- $\beta^* = +0.03$ rad $= +1.7°$: "Screen rotates clockwise (right edge closer)"

**Physical validation**:
1. Measure eye position relative to screen center (ruler, calipers)
2. Measure screen tilt with inclinometer or digital level
3. Compare measurements to estimates $\boldsymbol{\xi}_{\text{geo}}^*$
4. Expect agreement within 10-20 mm for position, 2-5° for angles

**Consistency checks**:
- Estimates should be stable across multiple calibrations with same setup
- If setup physically changed (screen moved), estimates should reflect this
- Within-session stability: Stage 1 vs. Stage 2 estimates similar

#### 10.7.10 Impact on Correction Quality

**Expected improvements**:

1. **Residual reduction**: 
   - Extended model typically reduces RMSE by 5-15%
   - Especially if misalignment is substantial ($|\Delta x|, |\Delta y| > 30$ mm or $|\alpha|, |\beta| > 5°$)

2. **Removal of systematic bias**: 
   - Corrected pupil $A_0(t)$ should show no residual correlation with gaze position
   - Check: $\text{Corr}(A_0^{\text{corrected}}, x_s) \approx 0$ and $\text{Corr}(A_0^{\text{corrected}}, y_s) \approx 0$

3. **Improved spatial consistency**: 
   - ICC (intra-class correlation) should increase
   - Target: ICC > 0.95

**When extended model helps most**:
- Naturalistic setups (laptops, home testing)
- Clinical populations with postural differences
- Large screens where positioning errors are common
- Multi-session studies (detects setup drift)

#### 10.7.11 Simplified Model Selection

**Question**: Is the added complexity (6 vs. 2 geometric parameters) justified?

**Model comparison**:
- Fit both simplified (Section 5) and extended (10.7) models
- Compare via Akaike Information Criterion (AIC) or Bayesian Information Criterion (BIC):

$$\text{AIC} = 2k + N \ln(\text{RSS}/N)$$

where $k$ is parameter count, $N$ is sample size, RSS is residual sum of squares.

**Decision rule**:
- $\Delta\text{AIC} > 10$: Extended model strongly preferred
- $2 < \Delta\text{AIC} < 10$: Extended model weakly preferred  
- $\Delta\text{AIC} < 2$: Models equivalent; use simpler

**Practical recommendation**: 
- Start with extended model (6 parameters)
- If estimates are small and uncertain ($|\Delta x|, |\Delta y| < 10$ mm, $|\alpha|, |\beta| < 2°$ with large SE), consider simplified model
- Most real setups benefit from extended model

#### 10.7.12 Binocular Extension with Alignment Parameters

For binocular tracking with nose-centered coordinates (Section 10.2):

**Modified geometry**:
- Nose at origin: $N = (0,0,0)$
- Eyes: $E_L = (-\text{IPD}/2, 0, 0)$, $E_R = (+\text{IPD}/2, 0, 0)$
- Screen center offset: $\vec{S}_0 = (\Delta x, \Delta y, d)$ relative to nose
- Screen tilt: $(\alpha, \beta)$ as before

**Eye-specific gaze vectors**:

$$\vec{E_iT}(x_s, y_s) = \vec{T}(x_s, y_s) - \vec{E_i}$$

where $\vec{T}(x_s, y_s)$ includes screen transformation (Section 10.7.3).

**Shared geometry parameters**: Both eyes constrain same $(\Delta x, \Delta y, \alpha, \beta)$

**Advantage**: Dual eyes provide independent information about screen geometry, improving parameter identifiability.

#### 10.7.13 Summary

**Extended model adds 4 geometric parameters**:
- Eye offset: $(\Delta x, \Delta y)$  
- Screen tilt: $(\alpha, \beta)$ for pitch and yaw
- Roll angle $\gamma$ omitted (negligible effect on foreshortening)

**Estimation via**:
- Joint optimization with temporal basis coefficients
- Soft priors (Gaussian or HalfNormal) for regularization
- Two-stage calibration maintains robustness

**Benefits**:
- Accounts for realistic experimental imperfections
- Improves correction accuracy (5-15% residual reduction)
- Provides interpretable diagnostic information
- Physically validatable estimates

**Computational cost**: Negligible (adds 4 parameters to existing optimization)

**Recommendation**: Use extended model by default for real experimental data.

## 11. References and Further Reading

**Foreshortening in pupillometry:**
- Hayes & Petrov (2016). Mapping and correcting the influence of gaze position on pupil size measurements. *Behavior Research Methods*
- Brisson et al. (2013). Pupil diameter measurement errors as a function of gaze direction in corneal reflection eyetrackers. *Behavior Research Methods*

**B-spline basis functions:**
- de Boor (1978). *A Practical Guide to Splines*. Springer.

**Optimization methods:**
- Nocedal & Wright (2006). *Numerical Optimization*. Springer.


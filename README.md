Field-Consistent Glyphs
=======================
**A ParaView filter for accurate and interpretable vector field visualization**

---

## Overview
Conventional arrow glyphs often fail to represent the true structure of curved or complex vector fields, leading to misleading visualizations. This project introduces **field-consistent glyphs**, a novel glyph-based method that traces streamlines both parallel and orthogonal to the field, generating arrow shapes that remain consistent with the local data across their entire geometry.

Developed as part of a software practical at Heidelberg University, the project consists of a custom **ParaView Python filter** that allows interactive and configurable visualization of 2D vector fields and experimental 3D tensor fields.

---

## Features
- **Field-Consistent glyphs:** Streamline-based arrows that follow both the direction and curvature of the field.
- **Multiple arrowhead variants:** Choose between smooth, staircase, or spline-based tips depending on your accuracy/performance needs.
- **Parameter controls:** Fine-tune step size, integration method (Euler/RK4), grid resolution, and more via ParaView's GUI.
- **Tensor field support (experimental):** Early work on 3D tensor glyphs using eigenvector decomposition.

---

## Installation
This filter is written in Python and intended for use as a plugin extending Heidelberg University's Visual Computing Group's "PRTL" extension.

Tested on:
- ParaView 5.x
- Ubuntu 18.04 (in VM)

---

## Usage
1. Load or create vector field data in ParaView (e.g. via PRTL's Python Model 2D).
2. Add the FieldConsistentGlyph Filter.
3. Set filter parameters in the GUI like:
    - Integration method (Euler/RK4)
    - Normalization toggle
    - Arrow length and width
    - Arrowhead style
4. Run the filter and view the glyphs. Adjust parameters for different visual effects or field clarity.

---

## Example fields
- **Radial Field:** ``F(x, y) = (x, y)``
- **Saddle Field:** ``F(x, y) = (x, -y)``
- **Rotational Field:** ``F(x, y) = (-y, x)``

Each highlights the advantage of using curved glyphs over conventional straight arrows.

---

## Limitations
- Higher computational cost per glyphs vs. standard arrows
- 3D tensor glyph visualization is only experimental
- Glyphs may self-intersect or stretch undesirably in high-curvature regions

---

## Future Work
- Adaptive glyph placement based on field topology
- Extension to full 3D vector and time-dependent fields
- Incorporating uncertainty visualization and user-driven interaction

---

## Credits
Developed by **David Hasse** as part of a software practical at Heidelberg University's Visual Computing Group.
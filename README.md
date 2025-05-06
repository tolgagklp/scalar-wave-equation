# Scalar Wave Equation Simulation

Modeling the scalar wave equation in both time and frequency domains — Software Lab, 2024.

## 🌀 Overview

Modelling the scalar wave equation in the time and frequency domain - Softwarelab 2024.

Wave simulation is used in many fields of modern-day engineering, such as biomedical applications, non-destructive testing (NDT), and Noise, Vibration and Harshness (NVH) tests. Different computational methods are applied to model the wave propagation. In the frame of this project, the finite difference method (FDM) is studied first, and then the finite element method (FEM) is introduced.

- **Finite Difference Method (FDM)** – Introduced first to model basic wave behavior.
- **Finite Element Method (FEM)** – Applied subsequently for more complex geometries and boundary conditions.

The simulation considers a **Isotropic and homogeneous** material domain.

> All code and related documentation developed as part of this group project can be found in this repository.

## 🛠️ Technologies Used

- Python (NumPy, Matplotlib, scipy)
- Finite Difference Method (FDM)
- Finite Element Method (FEM)
- Object-Oriented Programming (System, Node, etc.)

## 📁 Project Structure

```
scalar-wave-equation/
├── fdm/         # Finite Difference Method implementation
├── fem/         # Finite Element Method implementation
├── images/      # Visualizations, graphs, and wave plots
└── README.md
```

## 👥 Team Members

- Tolga Gökalp  
- Fatemeh Seyfi  
- Laura Winter  
- Julius Weidinger  

## 👩‍🏫 Supervisor

- Divya Singh

## 🖼️ Sample Visuals

You can include some example output visuals here, e.g.:

### Undamaged Material
![Wave propagation (undamaged)](images/undamaged.png)

### Damaged Material
![Wave propagation (damaged)](images/damaged.png)

### Frequency Response
![Frequency Response (1D)](images/frequency_domain_1d.png)

### Convergence
![Convergence](images/convergence.png)



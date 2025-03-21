# CS302Final

## Overview
This project explores the development of a Material Point Method (MPM) simulation locomotion using DiffTaichi, a differentiable physics simulation framework. By applying geometric modifications, optimization algorithms, and adaptive control strategies, a simulated soft-body structure that can move more efficiently and autonomously was developed. 

## Installation Requirements

Python 3.8+
Taichi
NumPy
Matplotlib

## Features
  Taichi-based Differentiable MPM: Implements MPM with automatic differentiation for optimization.
  Actuation and Control: Supports sinusoidal and feedback-based actuation for shape and locomotion design.
  Particle-to-Grid and Grid-to-Particle Transfers: Implements standard P2G and G2P mappings.
  Grid-Based Operations: Applies forces like gravity and handles boundary conditions.
  Optimization-Friendly: Enables backpropagation through physics simulations for learning-based control.

## Code Structure
  Initialization: Sets up Taichi fields and simulation parameters.
  Kernels:
    p2g(): Transfers particle properties to the grid.
    grid_op(): Updates grid velocities with forces and boundary conditions.
    g2p(): Transfers grid velocities back to particles.
    compute_actuation(): Generates actuation signals.
    compute_loss(): Computes differentiable loss for optimization.
  Scene Definition: Defines robots and particles with solid and actuator elements.
  Visualization: Uses Taichi GUI to render the simulation.

## Results & Observations

Structural Modifications: Improved stability and ground interaction.
Optimization: Enhanced forward motion by refining actuation parameters.
Control Strategies: Open-loop control enabled rhythmic movement, while closed-loop control allowed for adaptive responses to disturbances.
Computational Considerations: Balancing accuracy and efficiency was key to effective simulation performance.

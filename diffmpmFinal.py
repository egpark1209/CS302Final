import taichi as ti
import argparse
import os
import math
import numpy as np
import matplotlib.pyplot as plt

# --- Taichi Initialization ---
real = ti.f32
ti.init(default_fp=real, arch=ti.gpu, flatten_if=True)

# --- Simulation & Material Parameters ---
dim = 2
n_particles = 8192
n_solid_particles = 0
n_actuators = 0
n_grid = 128
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 1e-3
p_vol = 1
E = 10
mu = E
la = E
max_steps = 2048
steps = 2048
gravity = 3.8
target = [0.8, 0.2]

# --- Helper Lambdas for Field Definitions ---
scalar_field = lambda: ti.field(dtype=real)
vec_field = lambda: ti.Vector.field(dim, dtype=real)
mat_field = lambda: ti.Matrix.field(dim, dim, dtype=real)

# --- Field Definitions (Placed Later) ---
# These fields will be placed via ti.root.
actuator_id = ti.field(dtype=ti.i32)
particle_type = ti.field(dtype=ti.i32)
x, v = vec_field(), vec_field()
grid_v_in, grid_m_in = vec_field(), scalar_field()
grid_v_out = vec_field()
C, F = mat_field(), mat_field()

# Global fields defined with explicit shapes.
loss = ti.field(dtype=real, shape=())
avg_vx = ti.field(dtype=real, shape=())
x_avg = ti.Vector.field(dim, dtype=real, shape=())
control_mode = ti.field(dtype=ti.i32, shape=())

# Actuation parameters (topological/control design)
n_sin_waves = 4
weights = scalar_field()  # weights for sinusoidal actuation signals
bias = scalar_field()     # biases for actuation signals

# Actuation storage & parameters
actuation = scalar_field()  # stores computed actuation per time step and actuator
actuation_omega = 20
act_strength = 2.5

# --- Field Allocation ---
def allocate_fields():
    ti.root.dense(ti.ij, (n_actuators, n_sin_waves)).place(weights)
    ti.root.dense(ti.i, n_actuators).place(bias)
    ti.root.dense(ti.ij, (max_steps, n_actuators)).place(actuation)
    ti.root.dense(ti.i, n_particles).place(actuator_id, particle_type)
    ti.root.dense(ti.k, max_steps).dense(ti.l, n_particles).place(x, v, C, F)
    ti.root.dense(ti.ij, n_grid).place(grid_v_in, grid_m_in, grid_v_out)
    ti.root.lazy_grad()

# --- Grid and Particle Operations ---
@ti.kernel
def clear_grid():
    for i, j in grid_m_in:
        grid_v_in[i, j] = [0, 0]
        grid_m_in[i, j] = 0
        grid_v_in.grad[i, j] = [0, 0]
        grid_m_in.grad[i, j] = 0
        grid_v_out.grad[i, j] = [0, 0]

@ti.kernel
def clear_particle_grad():
    for f, i in x:
        x.grad[f, i] = [0, 0]
        v.grad[f, i] = [0, 0]
        C.grad[f, i] = [[0, 0], [0, 0]]
        F.grad[f, i] = [[0, 0], [0, 0]]

@ti.kernel
def clear_actuation_grad():
    for t, i in actuation:
        actuation[t, i] = 0.0

@ti.kernel
def p2g(f: ti.i32):
    # Particle-to-Grid: Transfer mass and momentum from particles to grid nodes.
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, ti.i32)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_F = (ti.Matrix.diag(dim=2, val=1) + dt * C[f, p]) @ F[f, p]
        J = new_F.determinant()
        if particle_type[p] == 0:  # fluid
            sqrtJ = ti.sqrt(J)
            new_F = ti.Matrix([[sqrtJ, 0], [0, sqrtJ]])
        F[f + 1, p] = new_F
        r, s = ti.polar_decompose(new_F)

        act_id = actuator_id[p]
        act = actuation[f, ti.max(0, act_id)] * act_strength
        if act_id == -1:
            act = 0.0

        A = ti.Matrix([[0.1, 0.0], [0.0, 1.0]]) * act
        cauchy = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
        mass = 0.0
        if particle_type[p] == 0:
            mass = 4
            cauchy = ti.Matrix([[1.0, 0.0], [0, 0.1]]) * (J - 1) * E
        else:
            mass = 1
            cauchy = 2 * mu * (new_F - r) @ new_F.transpose() + \
                     ti.Matrix.diag(2, la * (J - 1) * J)
        cauchy += new_F @ A @ new_F.transpose()
        stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
        affine = stress + mass * C[f, p]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                dpos = (ti.cast(ti.Vector([i, j]), real) - fx) * dx
                weight = w[i][0] * w[j][1]
                grid_v_in[base + offset] += weight * (mass * v[f, p] + affine @ dpos)
                grid_m_in[base + offset] += weight * mass

bound = 3
coeff = 0.5

@ti.kernel
def grid_op():
    # Grid operations: Apply gravity and boundary conditions.
    for i, j in grid_m_in:
        inv_m = 1 / (grid_m_in[i, j] + 1e-10)
        v_out = inv_m * grid_v_in[i, j]
        v_out[1] -= dt * gravity
        if i < bound and v_out[0] < 0:
            v_out[0] = 0
            v_out[1] = 0
        if i > n_grid - bound and v_out[0] > 0:
            v_out[0] = 0
            v_out[1] = 0
        if j < bound and v_out[1] < 0:
            v_out[0] = 0
            v_out[1] = 0
            normal = ti.Vector([0.0, 1.0])
            lsq = (normal**2).sum()
            if lsq > 0.5:
                if ti.static(coeff < 0):
                    v_out[0] = 0
                    v_out[1] = 0
                else:
                    lin = v_out.dot(normal)
                    if lin < 0:
                        vit = v_out - lin * normal
                        lit = v_out.norm() + 1e-10
                        if lit + coeff * lin <= 0:
                            v_out[0] = 0
                            v_out[1] = 0
                        else:
                            v_out = (1 + coeff * lin / lit) * vit
        if j > n_grid - bound and v_out[1] > 0:
            v_out[0] = 0
            v_out[1] = 0
        grid_v_out[i, j] = v_out

@ti.kernel
def g2p(f: ti.i32):
    # Grid-to-Particle: Transfer updated grid velocities back to particles.
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, real)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector([0.0, 0.0])
        new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                dpos = ti.cast(ti.Vector([i, j]), real) - fx
                g_v = grid_v_out[base[0] + i, base[1] + j]
                weight = w[i][0] * w[j][1]
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx
        v[f + 1, p] = new_v
        x[f + 1, p] = x[f, p] + dt * v[f + 1, p]
        C[f + 1, p] = new_C

# --- Actuation (Control) Kernels ---
@ti.kernel
def compute_actuation(t: ti.i32):
    # Open-Loop Control: Actuation is a time-based sinusoid.
    for i in range(n_actuators):
        act_val = 0.0
        for j in ti.static(range(n_sin_waves)):
            act_val += weights[i, j] * ti.sin(actuation_omega * t * dt +
                                            2 * math.pi / n_sin_waves * j)
        act_val += bias[i]
        actuation[t, i] = ti.tanh(act_val)

@ti.kernel
def compute_avg_velocity(t: ti.i32):
    # Compute the average x-velocity of solid particles at time step t.
    sum_vx = 0.0
    count = 0
    for p in range(n_particles):
        if particle_type[p] == 1:
            sum_vx += v[t, p][0]
            count += 1
    if count > 0:
        avg_vx[None] = sum_vx / count
    else:
        avg_vx[None] = 0.0

@ti.kernel
def compute_actuation_with_feedback(t: ti.i32):
    # Closed-Loop Control: Combine a sinusoidal (feedforward) signal with a feedback term.
    # The feedback here is a simple proportional controller based on the error between
    # a desired forward velocity and the current average x-velocity.
    for i in range(n_actuators):
        feedforward = 0.0
        for j in ti.static(range(n_sin_waves)):
            feedforward += weights[i, j] * ti.sin(actuation_omega * t * dt +
                                                  2 * math.pi / n_sin_waves * j)
        act_val = feedforward + bias[i] + 1.0 * (0.1 - avg_vx[None])
        actuation[t, i] = ti.tanh(act_val)

@ti.kernel
def compute_x_avg():
    # Compute the average x-position of all solid particles at the final step.
    for i in range(n_particles):
        contrib = 0.0
        if particle_type[i] == 1:
            contrib = 1.0 / n_solid_particles
        ti.atomic_add(x_avg[None], contrib * x[steps - 1, i])

@ti.kernel
def compute_loss():
    # Loss is defined as the negative of the average x-displacement.
    dist = x_avg[None][0]
    loss[None] = -dist

# --- Differentiable Time-Stepping ---
@ti.ad.grad_replaced
def advance(s):
    clear_grid()
    # Select control strategy based on control_mode:
    if control_mode[None] == 0:
        compute_actuation(s)
    else:
        compute_avg_velocity(s)
        compute_actuation_with_feedback(s)
    p2g(s)
    grid_op()
    g2p(s)

@ti.ad.grad_for(advance)
def advance_grad(s):
    clear_grid()
    if control_mode[None] == 0:
        compute_actuation.grad(s)
    else:
        compute_avg_velocity(s)
        compute_actuation_with_feedback.grad(s)
    p2g(s)
    grid_op()
    g2p.grad(s)
    grid_op.grad()
    p2g.grad(s)

def forward(total_steps=steps):
    # Run the simulation forward in time.
    for s in range(total_steps - 1):
        advance(s)
    x_avg[None] = [0, 0]
    compute_x_avg()
    compute_loss()

# --- Scene and Geometry Definitions ---
class Scene:
    def __init__(self):
        self.n_particles = 0
        self.n_solid_particles = 0
        self.x = []
        self.actuator_id = []
        self.particle_type = []
        self.offset_x = 0
        self.offset_y = 0

    def add_rect(self, x_coord, y_coord, w, h, actuation, ptype=1):
        if ptype == 0:
            assert actuation == -1
        global n_particles
        w_count = int(w / dx) * 2
        h_count = int(h / dx) * 2
        real_dx = w / w_count
        real_dy = h / h_count
        for i in range(w_count):
            for j in range(h_count):
                self.x.append([
                    x_coord + (i + 0.5) * real_dx + self.offset_x,
                    y_coord + (j + 0.5) * real_dy + self.offset_y
                ])
                self.actuator_id.append(actuation)
                self.particle_type.append(ptype)
                self.n_particles += 1
                self.n_solid_particles += int(ptype == 1)

    def add_oval(self, x_coord, y_coord, w, h, actuation, ptype=1):
        global n_particles
        a = w / 2 
        b = h / 2 
        center_x, center_y = x_coord + a, y_coord + b
        w_count = int(w / dx) * 2
        h_count = int(h / dx) * 2
        real_dx = w / w_count
        real_dy = h / h_count
        for i in range(w_count):
            for j in range(h_count):
                px = x_coord + (i + 0.5) * real_dx
                py = y_coord + (j + 0.5) * real_dy
                if ((px - center_x) ** 2) / (a ** 2) + ((py - center_y) ** 2) / (b ** 2) <= 1:
                    self.x.append([px + self.offset_x, py + self.offset_y])
                    self.actuator_id.append(actuation)
                    self.particle_type.append(ptype)
                    self.n_particles += 1
                    self.n_solid_particles += int(ptype == 1)

    def set_offset(self, x_val, y_val):
        self.offset_x = x_val
        self.offset_y = y_val

    def finalize(self):
        global n_particles, n_solid_particles
        n_particles = self.n_particles
        n_solid_particles = self.n_solid_particles
        print('n_particles', n_particles)
        print('n_solid', n_solid_particles)

    def set_n_actuators(self, n_act):
        global n_actuators
        n_actuators = n_act

def robot(scene): 
    # Define a robot geometry with a rectangular body and oval legs.
    scene.set_offset(0.1, 0.02)  # Raise the whole structure
    scene.add_rect(0.0, 0.2, 0.3, 0.1, -1) 
    leg_positions = [0.005, 0.08, 0.155, 0.23]  # x-positions for legs
    leg_y = 0.125
    leg_width = 0.05
    leg_height = 0.1
    for i, x_pos in enumerate(leg_positions):
        scene.add_oval(x_pos, leg_y, leg_width, leg_height, i)
    scene.set_n_actuators(4)

# --- Visualization ---
gui = ti.GUI("Differentiable MPM", (640, 640), background_color=0xFFFFFF)

def visualize(s, folder):
    aid = actuator_id.to_numpy()
    colors = np.empty(shape=n_particles, dtype=np.uint32)
    particles = x.to_numpy()[s]
    actuation_ = actuation.to_numpy()
    for i in range(n_particles):
        color = 0x111111
        if aid[i] != -1:
            act = actuation_[s - 1, int(aid[i])]
            color = ti.rgb_to_hex((0.5 - act, 0.5 - abs(act), 0.5 + act))
        colors[i] = color
    gui.circles(pos=particles, color=colors, radius=1.5)
    gui.line((0.05, 0.02), (0.95, 0.02), radius=3, color=0x0)
    os.makedirs(folder, exist_ok=True)
    gui.show(f'{folder}/{s:04d}.png')

# --- Main Optimization Loop ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--control_mode', type=int, default=0,
                        help="Control mode: 0 for open-loop, 1 for closed-loop (feedback)")
    options = parser.parse_args()

    # Initialize the scene and geometry.
    scene = Scene()
    robot(scene)
    scene.finalize()
    allocate_fields()   # Place all the remaining fields

    # Now that fields are placed, safely update control_mode.
    control_mode[None] = options.control_mode

    # Initialize actuation parameters (small random values).
    for i in range(n_actuators):
        for j in range(n_sin_waves):
            weights[i, j] = np.random.randn() * 0.01

    # Initialize particle states based on scene geometry.
    for i in range(scene.n_particles):
        x[0, i] = scene.x[i]
        F[0, i] = [[1, 0], [0, 1]]
        actuator_id[i] = scene.actuator_id[i]
        particle_type[i] = scene.particle_type[i]

    losses = []
    learning_rate = 0.1  # Learning rate for actuation parameter updates

    # Optimization loop: update actuation parameters to maximize forward displacement.
    for iter in range(options.iters):
        with ti.ad.Tape(loss):
            forward()
        l = loss[None]
        losses.append(l)
        print('Iteration:', iter, 'Loss:', l)

        # Gradient descent update for weights and bias.
        for i in range(n_actuators):
            for j in range(n_sin_waves):
                weights[i, j] -= learning_rate * weights.grad[i, j]
            bias[i] -= learning_rate * bias.grad[i]

        # (Optional) Visualize the simulation every 10 iterations.
        if iter % 10 == 0:
            forward(1500)
            for s in range(15, 1500, 16):
                visualize(s, f'diffmpm/iter{iter:03d}/')

    # Plotting: Plot loss (y-axis) against positive displacement (x-axis)
    # Note that since loss = - (positive displacement), we compute:
    positive_displacement = [-l for l in losses]
    plt.title("Loss vs. Positive Displacement")
    plt.xlabel("Positive Displacement (x-displacement)")
    plt.ylabel("Loss (negative x-displacement)")
    plt.plot(positive_displacement, losses, marker='o')
    plt.show()

if __name__ == '__main__':
    main()

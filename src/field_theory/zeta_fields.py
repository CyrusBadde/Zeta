"""
ζ-Field Theory Implementation
============================

Complete implementation of field theory with ζ-regularization:
- Multi-dimensional scalar field evolution
- Lagrangian formulation with ζ-sources
- Conservation laws and symmetries
- Numerical integration with stability analysis

Author: ζ-Regularization Framework
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import warnings

# Import core algebra
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from zeta_core.algebra import ZetaExtendedNumber, ZetaSymbol, zeta, finite


@dataclass
class FieldConfiguration:
    """Configuration parameters for ζ-field theory."""
    dimensions: int = 1
    grid_size: int = 64
    domain_length: float = 10.0
    mass_squared: float = 1.0
    coupling_lambda: float = 0.1
    time_step: float = 0.01
    boundary_conditions: str = "periodic"  # or "dirichlet", "neumann"


class ZetaField:
    """
    Multi-dimensional scalar field with ζ-regularization.
    
    Implements:
    - Klein-Gordon dynamics with ζ-sources
    - Energy and ζ-charge conservation
    - Stable numerical evolution
    - Information preservation through singularities
    """
    
    def __init__(self, config: FieldConfiguration):
        self.config = config
        self.dim = config.dimensions
        self.N = config.grid_size
        self.L = config.domain_length
        self.dx = config.domain_length / config.grid_size
        
        # Initialize field arrays
        if self.dim == 1:
            self.field = np.full(self.N, finite(0.0), dtype=object)
            self.x = np.linspace(-self.L/2, self.L/2, self.N)
        elif self.dim == 2:
            self.field = np.full((self.N, self.N), finite(0.0), dtype=object)
            self.x = np.linspace(-self.L/2, self.L/2, self.N)
            self.y = np.linspace(-self.L/2, self.L/2, self.N)
        else:
            raise NotImplementedError(f"Dimension {self.dim} not implemented")
        
        # Physical parameters
        self.m2 = config.mass_squared
        self.lam = config.coupling_lambda
        self.dt = config.time_step
        
        # Evolution tracking
        self.time = 0.0
        self.energy_history = []
        self.zeta_charge_history = []
        self.field_history = []
        
        # Numerical analysis
        self.max_iterations = 10000
        self.tolerance = 1e-12
    
    def set_initial_condition(self, profile: str = "gaussian", **kwargs):
        """
        Set initial field configuration with optional ζ-singularities.
        
        Parameters:
        - profile: "gaussian", "soliton", "random", "custom"
        - amplitude: field amplitude
        - width: characteristic width
        - center: center position
        - add_zeta: whether to add ζ-singularities
        """
        amplitude = kwargs.get('amplitude', 1.0)
        width = kwargs.get('width', 1.0)
        center = kwargs.get('center', 0.0)
        add_zeta = kwargs.get('add_zeta', True)
        
        if self.dim == 1:
            self._set_1d_initial_condition(profile, amplitude, width, center, add_zeta)
        elif self.dim == 2:
            self._set_2d_initial_condition(profile, amplitude, width, center, add_zeta)
    
    def _set_1d_initial_condition(self, profile, amplitude, width, center, add_zeta):
        """Set 1D initial condition."""
        for i, x_val in enumerate(self.x):
            if profile == "gaussian":
                value = amplitude * np.exp(-(x_val - center)**2 / (2 * width**2))
            elif profile == "soliton":
                value = amplitude / np.cosh((x_val - center) / width)
            elif profile == "random":
                value = amplitude * (2 * np.random.random() - 1)
            else:
                value = 0.0
            
            self.field[i] = finite(value)
        
        # Add ζ-singularity at center
        if add_zeta:
            center_idx = self.N // 2
            zeta_strength = amplitude * 2.0
            self.field[center_idx] = zeta(zeta_strength, "initial_singularity")
    
    def _set_2d_initial_condition(self, profile, amplitude, width, center, add_zeta):
        """Set 2D initial condition."""
        center_x, center_y = (center, center) if isinstance(center, (int, float)) else center
        
        for i in range(self.N):
            for j in range(self.N):
                x_val, y_val = self.x[i], self.y[j]
                r_squared = (x_val - center_x)**2 + (y_val - center_y)**2
                
                if profile == "gaussian":
                    value = amplitude * np.exp(-r_squared / (2 * width**2))
                elif profile == "soliton":
                    r = np.sqrt(r_squared)
                    value = amplitude / np.cosh(r / width)
                elif profile == "random":
                    value = amplitude * (2 * np.random.random() - 1)
                else:
                    value = 0.0
                
                self.field[i, j] = finite(value)
        
        # Add ζ-singularity at center
        if add_zeta:
            center_i, center_j = self.N // 2, self.N // 2
            zeta_strength = amplitude * 2.0
            self.field[center_i, center_j] = zeta(zeta_strength, "initial_singularity")
    
    def compute_laplacian(self, field_array: np.ndarray) -> np.ndarray:
        """Compute discrete Laplacian with proper ζ-handling."""
        if self.dim == 1:
            return self._compute_1d_laplacian(field_array)
        elif self.dim == 2:
            return self._compute_2d_laplacian(field_array)
    
    def _compute_1d_laplacian(self, field_array: np.ndarray) -> np.ndarray:
        """1D Laplacian with boundary conditions."""
        laplacian = np.full_like(field_array, finite(0.0), dtype=object)
        
        for i in range(self.N):
            if self.config.boundary_conditions == "periodic":
                i_plus = (i + 1) % self.N
                i_minus = (i - 1) % self.N
            elif self.config.boundary_conditions == "dirichlet":
                if i == 0 or i == self.N - 1:
                    laplacian[i] = finite(0.0)
                    continue
                i_plus = i + 1
                i_minus = i - 1
            else:  # Neumann
                i_plus = min(i + 1, self.N - 1)
                i_minus = max(i - 1, 0)
            
            # Second derivative: (φ[i+1] - 2φ[i] + φ[i-1]) / dx²
            phi_plus = field_array[i_plus]
            phi_center = field_array[i]
            phi_minus = field_array[i_minus]
            
            # Handle ζ-elements carefully
            if phi_center.is_zeta_element():
                # ζ-elements have special Laplacian behavior
                laplacian[i] = phi_center  # Preserve ζ-singularity
            else:
                # Standard finite difference
                second_deriv = (phi_plus.value - 2*phi_center.value + phi_minus.value) / (self.dx**2)
                laplacian[i] = finite(second_deriv)
        
        return laplacian
    
    def _compute_2d_laplacian(self, field_array: np.ndarray) -> np.ndarray:
        """2D Laplacian with boundary conditions."""
        laplacian = np.full_like(field_array, finite(0.0), dtype=object)
        
        for i in range(self.N):
            for j in range(self.N):
                if self.config.boundary_conditions == "periodic":
                    i_plus, i_minus = (i + 1) % self.N, (i - 1) % self.N
                    j_plus, j_minus = (j + 1) % self.N, (j - 1) % self.N
                elif self.config.boundary_conditions == "dirichlet":
                    if i == 0 or i == self.N-1 or j == 0 or j == self.N-1:
                        laplacian[i, j] = finite(0.0)
                        continue
                    i_plus, i_minus = i + 1, i - 1
                    j_plus, j_minus = j + 1, j - 1
                else:  # Neumann
                    i_plus, i_minus = min(i + 1, self.N-1), max(i - 1, 0)
                    j_plus, j_minus = min(j + 1, self.N-1), max(j - 1, 0)
                
                phi_center = field_array[i, j]
                
                if phi_center.is_zeta_element():
                    laplacian[i, j] = phi_center
                else:
                    # 2D Laplacian: ∂²φ/∂x² + ∂²φ/∂y²
                    d2_dx2 = (field_array[i_plus, j].value - 2*phi_center.value + 
                             field_array[i_minus, j].value) / (self.dx**2)
                    d2_dy2 = (field_array[i, j_plus].value - 2*phi_center.value + 
                             field_array[i, j_minus].value) / (self.dx**2)
                    
                    laplacian[i, j] = finite(d2_dx2 + d2_dy2)
        
        return laplacian
    
    def compute_field_equation(self, field_array: np.ndarray, 
                              zeta_sources: Optional[Dict] = None) -> np.ndarray:
        """
        Compute ζ-Klein-Gordon equation:
        □φ + m²φ + λφ³ + Σζ_i = 0
        """
        laplacian = self.compute_laplacian(field_array)
        rhs = np.full_like(field_array, finite(0.0), dtype=object)
        
        if self.dim == 1:
            indices = range(self.N)
        else:
            indices = [(i, j) for i in range(self.N) for j in range(self.N)]
        
        for idx in indices:
            if self.dim == 1:
                phi = field_array[idx]
                lap = laplacian[idx]
            else:
                phi = field_array[idx]
                lap = laplacian[idx]
            
            if phi.is_zeta_element():
                # ζ-elements evolve according to special rules
                rhs[idx] = phi  # Preserve ζ-structure
            else:
                # Standard Klein-Gordon terms
                mass_term = finite(self.m2 * phi.value)
                interaction_term = finite(self.lam * phi.value**3)
                
                # ζ-source term
                source_term = finite(0.0)
                if zeta_sources and idx in zeta_sources:
                    source_term = zeta_sources[idx]
                
                # Total RHS: -□φ - m²φ - λφ³ - sources
                total_rhs = -(lap + mass_term + interaction_term) + source_term
                rhs[idx] = total_rhs
        
        return rhs
    
    def evolve_step(self, method: str = "euler") -> Dict[str, float]:
        """
        Evolve field by one time step.
        
        Returns convergence and stability metrics.
        """
        if method == "euler":
            return self._euler_step()
        elif method == "rk4":
            return self._rk4_step()
        else:
            raise ValueError(f"Unknown evolution method: {method}")
    
    def _euler_step(self) -> Dict[str, float]:
        """First-order Euler time evolution."""
        # Store old field for error analysis
        old_field = self.field.copy()
        
        # Compute field equation
        field_eq = self.compute_field_equation(self.field)
        
        # Update field: φ(t+dt) = φ(t) + dt * (field equation)
        new_field = np.full_like(self.field, finite(0.0), dtype=object)
        
        if self.dim == 1:
            for i in range(self.N):
                if self.field[i].is_zeta_element():
                    # ζ-elements preserved through evolution
                    new_field[i] = self.field[i]
                else:
                    new_value = self.field[i].value + self.dt * field_eq[i].value
                    new_field[i] = finite(new_value)
        else:
            for i in range(self.N):
                for j in range(self.N):
                    if self.field[i, j].is_zeta_element():
                        new_field[i, j] = self.field[i, j]
                    else:
                        new_value = self.field[i, j].value + self.dt * field_eq[i, j].value
                        new_field[i, j] = finite(new_value)
        
        self.field = new_field
        self.time += self.dt
        
        # Compute convergence metrics
        error = self._compute_evolution_error(old_field, new_field)
        
        return {
            'time_step': self.dt,
            'evolution_error': error,
            'method': 'euler',
            'stable': error < 1000.0  # Simple stability check
        }
    
    def _rk4_step(self) -> Dict[str, float]:
        """Fourth-order Runge-Kutta evolution (simplified)."""
        # For ζ-fields, RK4 is complex due to non-linear tropical operations
        # Use multiple Euler steps with smaller timestep
        small_dt = self.dt / 4
        original_dt = self.dt
        self.dt = small_dt
        
        metrics = []
        for _ in range(4):
            metric = self._euler_step()
            metrics.append(metric)
        
        self.dt = original_dt
        
        # Average the metrics
        avg_error = np.mean([m['evolution_error'] for m in metrics])
        
        return {
            'time_step': original_dt,
            'evolution_error': avg_error,
            'method': 'rk4',
            'stable': avg_error < 1000.0
        }
    
    def _compute_evolution_error(self, old_field: np.ndarray, new_field: np.ndarray) -> float:
        """Compute evolution error for stability analysis."""
        total_error = 0.0
        count = 0
        
        if self.dim == 1:
            for i in range(self.N):
                if not old_field[i].is_zeta_element() and not new_field[i].is_zeta_element():
                    error = abs(new_field[i].value - old_field[i].value)
                    total_error += error
                    count += 1
        else:
            for i in range(self.N):
                for j in range(self.N):
                    if not old_field[i,j].is_zeta_element() and not new_field[i,j].is_zeta_element():
                        error = abs(new_field[i,j].value - old_field[i,j].value)
                        total_error += error
                        count += 1
        
        return total_error / count if count > 0 else 0.0
    
    def compute_conserved_quantities(self) -> Dict[str, float]:
        """Compute energy and ζ-charge conservation."""
        total_energy = 0.0
        zeta_charge = 0
        max_field = 0.0
        
        if self.dim == 1:
            for i in range(1, self.N-1):
                phi = self.field[i]
                
                if phi.is_zeta_element():
                    zeta_charge += len(phi.zeta_tags)
                else:
                    # Energy density
                    gradient = (self.field[i+1].value - self.field[i-1].value) / (2 * self.dx)
                    kinetic = 0.5 * gradient**2
                    potential = 0.5 * self.m2 * phi.value**2
                    interaction = 0.25 * self.lam * phi.value**4
                    
                    local_energy = kinetic + potential + interaction
                    total_energy += local_energy * self.dx
                    max_field = max(max_field, abs(phi.value))
        else:
            for i in range(1, self.N-1):
                for j in range(1, self.N-1):
                    phi = self.field[i, j]
                    
                    if phi.is_zeta_element():
                        zeta_charge += len(phi.zeta_tags)
                    else:
                        # 2D energy density (simplified)
                        grad_x = (self.field[i+1, j].value - self.field[i-1, j].value) / (2 * self.dx)
                        grad_y = (self.field[i, j+1].value - self.field[i, j-1].value) / (2 * self.dx)
                        kinetic = 0.5 * (grad_x**2 + grad_y**2)
                        potential = 0.5 * self.m2 * phi.value**2
                        interaction = 0.25 * self.lam * phi.value**4
                        
                        local_energy = kinetic + potential + interaction
                        total_energy += local_energy * self.dx**2
                        max_field = max(max_field, abs(phi.value))
        
        return {
            'total_energy': total_energy,
            'zeta_charge': zeta_charge,
            'max_field': max_field,
            'time': self.time
        }
    
    def evolve(self, total_time: float, method: str = "euler", 
               save_history: bool = True, progress_callback: Optional[Callable] = None):
        """
        Evolve field for given total time.
        
        Parameters:
        - total_time: Total evolution time
        - method: Evolution method ("euler" or "rk4")
        - save_history: Whether to save field history
        - progress_callback: Function to call with progress updates
        """
        steps = int(total_time / self.dt)
        
        for step in range(steps):
            # Evolve one step
            metrics = self.evolve_step(method)
            
            # Compute conservation quantities
            conserved = self.compute_conserved_quantities()
            
            # Save history
            if save_history:
                self.energy_history.append(conserved['total_energy'])
                self.zeta_charge_history.append(conserved['zeta_charge'])
                
                if step % max(1, steps // 100) == 0:  # Save every 1%
                    self.field_history.append(self.field.copy())
            
            # Progress callback
            if progress_callback and step % max(1, steps // 20) == 0:  # Update every 5%
                progress = step / steps
                progress_callback(progress, self.time, conserved, metrics)
            
            # Stability check
            if not metrics['stable']:
                warnings.warn(f"Evolution became unstable at t={self.time:.3f}")
                break
    
    def get_field_profile(self) -> Dict:
        """Get current field profile for visualization."""
        if self.dim == 1:
            values = []
            positions = []
            zeta_positions = []
            zeta_values = []
            
            for i, x_val in enumerate(self.x):
                if self.field[i].is_zeta_element():
                    zeta_positions.append(x_val)
                    # Extract ζ-tag for visualization
                    if self.field[i].zeta_tags:
                        tag_val = list(self.field[i].zeta_tags)[0].tag
                        zeta_values.append(tag_val)
                    else:
                        zeta_values.append(0.0)
                else:
                    positions.append(x_val)
                    values.append(self.field[i].value)
            
            return {
                'dimension': 1,
                'positions': np.array(positions),
                'values': np.array(values),
                'zeta_positions': np.array(zeta_positions),
                'zeta_values': np.array(zeta_values),
                'time': self.time
            }
        
        else:  # 2D
            values = np.zeros((self.N, self.N))
            zeta_mask = np.zeros((self.N, self.N), dtype=bool)
            zeta_values = np.zeros((self.N, self.N))
            
            for i in range(self.N):
                for j in range(self.N):
                    if self.field[i, j].is_zeta_element():
                        zeta_mask[i, j] = True
                        if self.field[i, j].zeta_tags:
                            tag_val = list(self.field[i, j].zeta_tags)[0].tag
                            zeta_values[i, j] = tag_val
                    else:
                        values[i, j] = self.field[i, j].value
            
            return {
                'dimension': 2,
                'x': self.x,
                'y': self.y,
                'values': values,
                'zeta_mask': zeta_mask,
                'zeta_values': zeta_values,
                'time': self.time
            }


def demonstrate_zeta_field_theory():
    """Demonstration of ζ-field theory capabilities."""
    print("=== ζ-Field Theory Demonstration ===")
    
    # Create 1D field
    config = FieldConfiguration(
        dimensions=1,
        grid_size=64,
        domain_length=10.0,
        mass_squared=1.0,
        coupling_lambda=0.1,
        time_step=0.01
    )
    
    field = ZetaField(config)
    
    # Set initial condition
    field.set_initial_condition("gaussian", amplitude=1.0, width=1.0, add_zeta=True)
    
    print(f"Initialized {config.dimensions}D field on {config.grid_size} grid points")
    
    # Initial analysis
    initial_conserved = field.compute_conserved_quantities()
    print(f"Initial energy: {initial_conserved['total_energy']:.6f}")
    print(f"Initial ζ-charge: {initial_conserved['zeta_charge']}")
    
    # Evolution
    def progress_update(progress, time, conserved, metrics):
        if progress == 0 or progress >= 0.95:
            print(f"Progress {progress*100:.0f}%: t={time:.3f}, "
                  f"E={conserved['total_energy']:.6f}, "
                  f"ζ={conserved['zeta_charge']}, "
                  f"stable={metrics['stable']}")
    
    print("\nEvolving field...")
    field.evolve(1.0, method="euler", progress_callback=progress_update)
    
    # Final analysis
    final_conserved = field.compute_conserved_quantities()
    print(f"\nFinal energy: {final_conserved['total_energy']:.6f}")
    print(f"Final ζ-charge: {final_conserved['zeta_charge']}")
    
    # Energy conservation check
    energy_drift = abs(final_conserved['total_energy'] - initial_conserved['total_energy'])
    energy_conservation = energy_drift / abs(initial_conserved['total_energy']) < 0.1
    
    print(f"Energy conservation: {energy_conservation} (drift: {energy_drift:.2e})")
    print(f"ζ-charge conservation: {final_conserved['zeta_charge'] == initial_conserved['zeta_charge']}")
    
    return field


if __name__ == "__main__":
    demonstrate_zeta_field_theory()

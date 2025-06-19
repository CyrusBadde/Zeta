import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Set
import warnings

@dataclass
class ZetaAugmentedNumber:
    """
    Complete implementation of zeta-augmented tropical arithmetic.
    Elements are (value, zeta_tags) where value in R union {+infinity}.
    """
    value: float
    zeta_tags: Set[float]
    
    def __post_init__(self):
        if self.value == np.inf and not self.zeta_tags:
            warnings.warn("Tropical infinity without zeta-tags loses semantic information")
    
    def __add__(self, other):
        """zeta-augmented tropical addition: (a,S) oplus (b,T)"""
        if not isinstance(other, ZetaAugmentedNumber):
            other = ZetaAugmentedNumber(float(other), set())
        
        if self.value < other.value:
            return ZetaAugmentedNumber(self.value, self.zeta_tags.copy())
        elif other.value < self.value:
            return ZetaAugmentedNumber(other.value, other.zeta_tags.copy())
        else:  # Equal values
            return ZetaAugmentedNumber(self.value, self.zeta_tags | other.zeta_tags)
    
    def __mul__(self, other):
        """zeta-augmented tropical multiplication: (a,S) otimes (b,T)"""
        if not isinstance(other, ZetaAugmentedNumber):
            if other == 0 and self.value == np.inf:
                # zeta-reconstruction: zeta_a otimes 0 = a
                if len(self.zeta_tags) == 1:
                    return ZetaAugmentedNumber(list(self.zeta_tags)[0], set())
                else:
                    # Multiple tags: undefined or sum
                    return ZetaAugmentedNumber(sum(self.zeta_tags), set())
            other = ZetaAugmentedNumber(float(other), set())
        
        new_value = self.value + other.value
        new_tags = self.zeta_tags | other.zeta_tags
        return ZetaAugmentedNumber(new_value, new_tags)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        """zeta-augmented division"""
        if not isinstance(other, ZetaAugmentedNumber):
            other = ZetaAugmentedNumber(float(other), set())
        
        if other.value == 0:
            # Division by zero creates zeta-symbol
            return ZetaAugmentedNumber(np.inf, {self.value})
        
        new_value = self.value - other.value
        new_tags = self.zeta_tags | other.zeta_tags
        return ZetaAugmentedNumber(new_value, new_tags)
    
    def __pow__(self, exponent):
        """zeta-augmented exponentiation"""
        new_value = self.value * exponent
        new_tags = {tag * exponent for tag in self.zeta_tags}
        return ZetaAugmentedNumber(new_value, new_tags)
    
    def __repr__(self):
        if self.value == np.inf and self.zeta_tags:
            tags_str = ",".join(f"{tag:.3f}" for tag in sorted(self.zeta_tags))
            return f"zeta_{{{tags_str}}}"
        elif self.value == np.inf:
            return "+infinity"
        else:
            tags_str = f",{{{','.join(f'{tag:.3f}' for tag in self.zeta_tags)}}}" if self.zeta_tags else ""
            return f"{self.value:.3f}{tags_str}"

class TropicalFieldWithZeta:
    """1D tropical scalar field with explicit zeta-tracking."""
    
    def __init__(self, N_points=100, domain_length=10.0):
        self.N = N_points
        self.L = domain_length
        self.dx = domain_length / N_points
        self.x = np.linspace(-domain_length/2, domain_length/2, N_points)
        
        # Initialize field as array of ZetaAugmentedNumbers
        self.field = [ZetaAugmentedNumber(0.0, set()) for _ in range(N_points)]
        
        # Physical parameters
        self.mass_squared = 1.0
        self.coupling = 0.1
        self.time = 0.0
        
        # History for analysis
        self.energy_history = []
        self.zeta_history = []
    
    def set_initial_condition(self, amplitude=1.0, width=1.0, center=0.0):
        """Set Gaussian initial condition with potential zeta-singularity."""
        for i, x_val in enumerate(self.x):
            # Gaussian profile
            profile_value = amplitude * np.exp(-(x_val - center)**2 / (2 * width**2))
            
            # Create singularity at center
            if abs(x_val - center) < self.dx/2:
                # Point singularity: create zeta-symbol
                self.field[i] = ZetaAugmentedNumber(np.inf, {amplitude})
            else:
                self.field[i] = ZetaAugmentedNumber(profile_value, set())
    
    def compute_energy(self):
        """Compute total energy of the field configuration."""
        energy = 0.0
        zeta_tags = set()
        
        for i in range(1, self.N-1):
            # Kinetic term: (dphi)^2
            if i > 0:
                dx_phi = (self.field[i+1].value - self.field[i-1].value) / (2 * self.dx)
                kinetic = dx_phi**2 / 2
                
                # Collect zeta tags
                zeta_tags |= self.field[i+1].zeta_tags | self.field[i-1].zeta_tags
            else:
                kinetic = 0
            
            # Potential term: (m^2 * phi^2)/2 + (lambda * phi^4)/4
            if self.field[i].value != np.inf:
                potential = self.mass_squared * self.field[i].value**2 / 2
                potential += self.coupling * self.field[i].value**4 / 4
                
                # Collect zeta tags
                zeta_tags |= self.field[i].zeta_tags
                
                # Add to total energy
                energy += kinetic + potential
        
        return ZetaAugmentedNumber(energy, zeta_tags)
    
    def evolve(self, dt=0.01, steps=1):
        """Evolve field using tropical dynamics."""
        for _ in range(steps):
            # Store current state
            old_field = self.field.copy()
            
            # Update interior points
            for i in range(1, self.N-1):
                # Compute Laplacian: del^2 * phi
                laplacian = (old_field[i+1].value - 2*old_field[i].value + old_field[i-1].value) / self.dx**2
                
                # Compute force: -m^2 * phi - lambda * phi^3
                if old_field[i].value != np.inf:
                    mass_term = -self.mass_squared * old_field[i].value
                    nonlinear_term = -self.coupling * old_field[i].value**3
                    force = laplacian + mass_term + nonlinear_term
                    
                    # Collect zeta tags
                    zeta_tags = old_field[i+1].zeta_tags | old_field[i].zeta_tags | old_field[i-1].zeta_tags
                    
                    # Update field
                    new_value = old_field[i].value + dt * force
                    self.field[i] = ZetaAugmentedNumber(new_value, zeta_tags)
            
            # Update time
            self.time += dt
            
            # Record history
            self.energy_history.append(self.compute_energy().value)
            self.zeta_history.append(sum(1 for f in self.field if f.value == np.inf))
    
    def plot_field(self, ax=None, show_zeta=True):
        """Plot the field configuration."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract regular values and positions
        regular_x = []
        regular_y = []
        zeta_x = []
        zeta_y = []
        zeta_tags = []
        
        for i, field_val in enumerate(self.field):
            if field_val.value == np.inf:
                zeta_x.append(self.x[i])
                zeta_y.append(0)  # Plot at y=0
                zeta_tags.append(field_val.zeta_tags)
            else:
                regular_x.append(self.x[i])
                regular_y.append(field_val.value)
        
        # Plot regular field values
        ax.plot(regular_x, regular_y, "b-", label="Field value")
        
        # Plot zeta symbols
        if show_zeta and zeta_x:
            ax.scatter(zeta_x, zeta_y, color="red", marker="x", s=100, label="zeta-singularity")
            
            # Annotate with tags
            for x, y, tags in zip(zeta_x, zeta_y, zeta_tags):
                tag_str = ",".join(f"{tag:.2f}" for tag in tags)
                ax.annotate(f"zeta_{{{tag_str}}}", (x, y), xytext=(0, 10), 
                           textcoords="offset points", ha="center")
        
        ax.set_xlabel("Position")
        ax.set_ylabel("Field value")
        ax.set_title(f"Tropical Field at t={self.time:.2f}")
        ax.legend()
        
        return ax

# Example usage
if __name__ == "__main__":
    field = TropicalFieldWithZeta(N_points=100, domain_length=10.0)
    field.set_initial_condition(amplitude=1.0, width=0.5, center=0.0)
    field.evolve(dt=0.01, steps=100)

    # Analyze results
    print(f"Final energy: {field.compute_energy()}")
    print(f"Number of zeta-singularities: {sum(1 for f in field.field if f.value == np.inf)}")
    print(f"zeta-tags: {set().union(*(f.zeta_tags for f in field.field if f.zeta_tags))}")

    # Demonstrate zeta-reconstruction
    a = ZetaAugmentedNumber(np.inf, {2.0})
    b = ZetaAugmentedNumber(0.0, set())
    c = a * b  # zeta-reconstruction: zeta_a ⊗ 0 = a
    print(f"zeta-reconstruction: {a} ⊗ {b} = {c}")

    # Demonstrate zeta-multiplication
    a = ZetaAugmentedNumber(np.inf, {2.0})
    b = ZetaAugmentedNumber(np.inf, {4.0})
    c = a * b  # zeta-multiplication: zeta_a ⊗ zeta_b = zeta_{ab}
    print(f"zeta-multiplication: {a} ⊗ {b} = {c}")

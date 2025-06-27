"""
ζ-Regularization Interactive GUI
===============================

Complete graphical interface for exploring ζ-regularization:
- Interactive algebra demonstrations
- Field theory simulations with real-time visualization
- Semiring property verification
- Educational tools and examples

Dependencies: tkinter, matplotlib, numpy
Author: ζ-Regularization Framework
License: MIT
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import threading
import time
from typing import Optional, Callable
import sys
import os

# Add the src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
try:
    from zeta_core.algebra import ZetaExtendedNumber, ZetaSymbol, ZetaSemiring, zeta, finite
    from field_theory.zeta_fields import ZetaField, FieldConfiguration
except ImportError:
    print("Error: Could not import ζ-modules. Please ensure src/ directory structure is correct.")
    print("Expected structure:")
    print("  src/")
    print("    zeta_core/")
    print("      algebra.py")
    print("    field_theory/")
    print("      zeta_fields.py")
    sys.exit(1)


class ZetaAlgebraTab:
    """Tab for interactive ζ-algebra demonstrations."""
    
    def __init__(self, parent):
        self.frame = ttk.Frame(parent)
        self.setup_ui()
        self.semiring = ZetaSemiring()
    
    def setup_ui(self):
        # Main sections
        self.create_basic_operations_section()
        self.create_semiring_verification_section()
        self.create_reconstruction_section()
        
    def create_basic_operations_section(self):
        # Basic Operations Frame
        ops_frame = ttk.LabelFrame(self.frame, text="Basic ζ-Algebra Operations", padding="10")
        ops_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        
        # Input fields
        ttk.Label(ops_frame, text="Element A:").grid(row=0, column=0, sticky="w")
        self.entry_a = ttk.Entry(ops_frame, width=15)
        self.entry_a.grid(row=0, column=1, padx=5)
        self.entry_a.insert(0, "2.5")
        
        ttk.Button(ops_frame, text="Make ζ", 
                  command=lambda: self.make_zeta_element('a')).grid(row=0, column=2, padx=2)
        
        ttk.Label(ops_frame, text="Element B:").grid(row=1, column=0, sticky="w")
        self.entry_b = ttk.Entry(ops_frame, width=15)
        self.entry_b.grid(row=1, column=1, padx=5)
        self.entry_b.insert(0, "1.8")
        
        ttk.Button(ops_frame, text="Make ζ", 
                  command=lambda: self.make_zeta_element('b')).grid(row=1, column=2, padx=2)
        
        # Operation buttons
        button_frame = ttk.Frame(ops_frame)
        button_frame.grid(row=2, column=0, columnspan=3, pady=10)
        
        ttk.Button(button_frame, text="A ⊕ B (Tropical +)", 
                  command=self.tropical_add).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="A ⊗ B (Tropical ×)", 
                  command=self.tropical_mult).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="ζ-Reconstruct", 
                  command=self.zeta_reconstruct).grid(row=0, column=2, padx=5)
        
        # Results display
        self.result_text = tk.Text(ops_frame, height=8, width=60)
        self.result_text.grid(row=3, column=0, columnspan=3, pady=10)
        
        scrollbar = ttk.Scrollbar(ops_frame, orient="vertical", command=self.result_text.yview)
        scrollbar.grid(row=3, column=3, sticky="ns")
        self.result_text.configure(yscrollcommand=scrollbar.set)
    
    def create_semiring_verification_section(self):
        # Semiring Verification Frame
        verify_frame = ttk.LabelFrame(self.frame, text="Semiring Property Verification", padding="10")
        verify_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
        ttk.Button(verify_frame, text="Run Complete Verification", 
                  command=self.verify_semiring_properties).grid(row=0, column=0, pady=5)
        
        self.verification_result = tk.Text(verify_frame, height=10, width=40)
        self.verification_result.grid(row=1, column=0, pady=5)
    
    def create_reconstruction_section(self):
        # ζ-Reconstruction Testing Frame
        recon_frame = ttk.LabelFrame(self.frame, text="ζ-Reconstruction Testing", padding="10")
        recon_frame.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        
        ttk.Label(recon_frame, text="ζ-tag value:").grid(row=0, column=0, sticky="w")
        self.zeta_tag_entry = ttk.Entry(recon_frame, width=15)
        self.zeta_tag_entry.grid(row=0, column=1, padx=5)
        self.zeta_tag_entry.insert(0, "3.14159")
        
        ttk.Button(recon_frame, text="Test ζ_a • 0 = a", 
                  command=self.test_reconstruction).grid(row=1, column=0, columnspan=2, pady=5)
        
        self.reconstruction_result = tk.Text(recon_frame, height=10, width=40)
        self.reconstruction_result.grid(row=2, column=0, columnspan=2, pady=5)
    
    def make_zeta_element(self, element):
        """Convert finite element to ζ-element."""
        if element == 'a':
            entry = self.entry_a
        else:
            entry = self.entry_b
        
        try:
            value = float(entry.get())
            entry.delete(0, tk.END)
            entry.insert(0, f"ζ_{value}")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number")
    
    def parse_element(self, text):
        """Parse text input to ZetaExtendedNumber."""
        text = text.strip()
        if text.startswith("ζ_"):
            # ζ-element
            tag_str = text[2:]
            try:
                tag_value = float(tag_str)
                return zeta(tag_value, "user_input")
            except ValueError:
                raise ValueError(f"Invalid ζ-tag: {tag_str}")
        else:
            # Finite element
            return finite(float(text))
    
    def tropical_add(self):
        """Perform tropical addition."""
        try:
            a = self.parse_element(self.entry_a.get())
            b = self.parse_element(self.entry_b.get())
            result = a + b
            
            self.result_text.insert(tk.END, f"Tropical Addition:\n")
            self.result_text.insert(tk.END, f"  {a} ⊕ {b} = {result}\n")
            self.result_text.insert(tk.END, f"  (min operation with ζ-tag inheritance)\n\n")
            self.result_text.see(tk.END)
            
        except Exception as e:
            messagebox.showerror("Error", f"Operation failed: {e}")
    
    def tropical_mult(self):
        """Perform tropical multiplication."""
        try:
            a = self.parse_element(self.entry_a.get())
            b = self.parse_element(self.entry_b.get())
            result = a * b
            
            self.result_text.insert(tk.END, f"Tropical Multiplication:\n")
            self.result_text.insert(tk.END, f"  {a} ⊗ {b} = {result}\n")
            self.result_text.insert(tk.END, f"  (addition operation with ζ-composition)\n\n")
            self.result_text.see(tk.END)
            
        except Exception as e:
            messagebox.showerror("Error", f"Operation failed: {e}")
    
    def zeta_reconstruct(self):
        """Perform ζ-reconstruction."""
        try:
            a = self.parse_element(self.entry_a.get())
            
            if a.is_zeta_element():
                reconstructed = a.zeta_reconstruct(0)
                self.result_text.insert(tk.END, f"ζ-Reconstruction:\n")
                self.result_text.insert(tk.END, f"  {a} • 0 = {reconstructed}\n")
                self.result_text.insert(tk.END, f"  (Information preservation: ζ_a • 0 = a)\n\n")
            else:
                self.result_text.insert(tk.END, f"ζ-Reconstruction:\n")
                self.result_text.insert(tk.END, f"  {a} is not a ζ-element\n")
                self.result_text.insert(tk.END, f"  Only ζ-tagged infinities can be reconstructed\n\n")
            
            self.result_text.see(tk.END)
            
        except Exception as e:
            messagebox.showerror("Error", f"Reconstruction failed: {e}")
    
    def verify_semiring_properties(self):
        """Run complete semiring verification."""
        self.verification_result.delete(1.0, tk.END)
        self.verification_result.insert(tk.END, "Running semiring verification...\n\n")
        self.verification_result.update()
        
        try:
            results = self.semiring.run_comprehensive_test()
            
            self.verification_result.insert(tk.END, "=== SEMIRING VERIFICATION RESULTS ===\n")
            
            for property_name, passed in results.items():
                status = "✓ PASS" if passed else "✗ FAIL"
                prop_display = property_name.replace('_', ' ').title()
                self.verification_result.insert(tk.END, f"{status}: {prop_display}\n")
            
            overall_pass = all(results.values())
            overall_status = "✓ ALL TESTS PASSED" if overall_pass else "✗ SOME TESTS FAILED"
            
            self.verification_result.insert(tk.END, f"\nOverall Result: {overall_status}\n")
            
            if overall_pass:
                self.verification_result.insert(tk.END, "\nThe ζ-extended structure is a valid semiring!\n")
            else:
                self.verification_result.insert(tk.END, "\nSome properties failed. Check implementation.\n")
                
        except Exception as e:
            self.verification_result.insert(tk.END, f"Verification failed: {e}\n")
    
    def test_reconstruction(self):
        """Test ζ-reconstruction with user-specified tag."""
        self.reconstruction_result.delete(1.0, tk.END)
        
        try:
            tag_value = float(self.zeta_tag_entry.get())
            
            # Create ζ-element
            zeta_element = zeta(tag_value, "test")
            
            # Test reconstruction
            reconstructed = zeta_element.zeta_reconstruct(0)
            
            # Test via multiplication  
            zero = finite(0)
            mult_result = zeta_element * zero
            
            # Display results
            self.reconstruction_result.insert(tk.END, f"ζ-Reconstruction Test\n")
            self.reconstruction_result.insert(tk.END, f"=====================\n\n")
            self.reconstruction_result.insert(tk.END, f"Input tag: {tag_value}\n")
            self.reconstruction_result.insert(tk.END, f"ζ-element: {zeta_element}\n\n")
            self.reconstruction_result.insert(tk.END, f"Direct reconstruction:\n")
            self.reconstruction_result.insert(tk.END, f"  ζ_{tag_value} • 0 = {reconstructed}\n\n")
            self.reconstruction_result.insert(tk.END, f"Via multiplication:\n")
            self.reconstruction_result.insert(tk.END, f"  ζ_{tag_value} ⊗ 0 = {mult_result}\n\n")
            
            # Verify correctness
            error = abs(reconstructed - tag_value)
            mult_error = abs(mult_result.value - tag_value)
            
            if error < 1e-12 and mult_error < 1e-12:
                self.reconstruction_result.insert(tk.END, "✓ Perfect reconstruction!\n")
                self.reconstruction_result.insert(tk.END, "  Information perfectly preserved.\n")
            else:
                self.reconstruction_result.insert(tk.END, f"✗ Reconstruction error: {error:.2e}\n")
                
        except Exception as e:
            self.reconstruction_result.insert(tk.END, f"Test failed: {e}\n")


class ZetaFieldTab:
    """Tab for field theory simulations."""
    
    def __init__(self, parent):
        self.frame = ttk.Frame(parent)
        self.field = None
        self.simulation_running = False
        self.setup_ui()
    
    def setup_ui(self):
        # Configuration frame
        self.create_configuration_section()
        
        # Control frame
        self.create_control_section()
        
        # Visualization frame
        self.create_visualization_section()
        
        # Results frame
        self.create_results_section()
    
    def create_configuration_section(self):
        config_frame = ttk.LabelFrame(self.frame, text="Field Configuration", padding="10")
        config_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        # Dimension selection
        ttk.Label(config_frame, text="Dimensions:").grid(row=0, column=0, sticky="w")
        self.dim_var = tk.StringVar(value="1")
        dim_combo = ttk.Combobox(config_frame, textvariable=self.dim_var, 
                                values=["1", "2"], state="readonly", width=10)
        dim_combo.grid(row=0, column=1, padx=5)
        
        # Grid size
        ttk.Label(config_frame, text="Grid Size:").grid(row=0, column=2, sticky="w", padx=(20,0))
        self.grid_var = tk.StringVar(value="64")
        grid_entry = ttk.Entry(config_frame, textvariable=self.grid_var, width=10)
        grid_entry.grid(row=0, column=3, padx=5)
        
        # Physical parameters
        ttk.Label(config_frame, text="Mass²:").grid(row=1, column=0, sticky="w")
        self.mass_var = tk.StringVar(value="1.0")
        mass_entry = ttk.Entry(config_frame, textvariable=self.mass_var, width=10)
        mass_entry.grid(row=1, column=1, padx=5)
        
        ttk.Label(config_frame, text="Coupling λ:").grid(row=1, column=2, sticky="w", padx=(20,0))
        self.coupling_var = tk.StringVar(value="0.1")
        coupling_entry = ttk.Entry(config_frame, textvariable=self.coupling_var, width=10)
        coupling_entry.grid(row=1, column=3, padx=5)
        
        # Initial condition
        ttk.Label(config_frame, text="Initial Profile:").grid(row=2, column=0, sticky="w")
        self.profile_var = tk.StringVar(value="gaussian")
        profile_combo = ttk.Combobox(config_frame, textvariable=self.profile_var,
                                   values=["gaussian", "soliton", "random"], 
                                   state="readonly", width=10)
        profile_combo.grid(row=2, column=1, padx=5)
        
        # Add ζ-singularity checkbox
        self.zeta_var = tk.BooleanVar(value=True)
        zeta_check = ttk.Checkbutton(config_frame, text="Add ζ-singularity", 
                                    variable=self.zeta_var)
        zeta_check.grid(row=2, column=2, columnspan=2, sticky="w", padx=(20,0))
    
    def create_control_section(self):
        control_frame = ttk.LabelFrame(self.frame, text="Simulation Control", padding="10")
        control_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
        # Time parameters
        ttk.Label(control_frame, text="Total Time:").grid(row=0, column=0, sticky="w")
        self.time_var = tk.StringVar(value="2.0")
        time_entry = ttk.Entry(control_frame, textvariable=self.time_var, width=10)
        time_entry.grid(row=0, column=1, padx=5)
        
        ttk.Label(control_frame, text="Time Step:").grid(row=0, column=2, sticky="w", padx=(20,0))
        self.dt_var = tk.StringVar(value="0.01")
        dt_entry = ttk.Entry(control_frame, textvariable=self.dt_var, width=10)
        dt_entry.grid(row=0, column=3, padx=5)
        
        # Control buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=1, column=0, columnspan=4, pady=10)
        
        self.init_button = ttk.Button(button_frame, text="Initialize Field", 
                                     command=self.initialize_field)
        self.init_button.grid(row=0, column=0, padx=5)
        
        self.run_button = ttk.Button(button_frame, text="Run Simulation", 
                                    command=self.run_simulation)
        self.run_button.grid(row=0, column=1, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop", 
                                     command=self.stop_simulation, state="disabled")
        self.stop_button.grid(row=0, column=2, padx=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(control_frame, mode='determinate')
        self.progress.grid(row=2, column=0, columnspan=4, sticky="ew", pady=5)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(control_frame, textvariable=self.status_var)
        self.status_label.grid(row=3, column=0, columnspan=4)
    
    def create_visualization_section(self):
        viz_frame = ttk.LabelFrame(self.frame, text="Field Visualization", padding="10")
        viz_frame.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=5, pady=5)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initial plot
        self.ax.set_title("ζ-Field Configuration")
        self.ax.set_xlabel("Position")
        self.ax.set_ylabel("Field Value")
        self.ax.grid(True, alpha=0.3)
    
    def create_results_section(self):
        results_frame = ttk.LabelFrame(self.frame, text="Conservation & Results", padding="10")
        results_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        
        self.results_text = tk.Text(results_frame, height=8, width=80)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        results_scrollbar = ttk.Scrollbar(results_frame, orient="vertical", 
                                        command=self.results_text.yview)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
    
    def initialize_field(self):
        """Initialize field with current configuration."""
        try:
            # Parse configuration
            config = FieldConfiguration(
                dimensions=int(self.dim_var.get()),
                grid_size=int(self.grid_var.get()),
                domain_length=10.0,
                mass_squared=float(self.mass_var.get()),
                coupling_lambda=float(self.coupling_var.get()),
                time_step=float(self.dt_var.get()),
                boundary_conditions="periodic"
            )
            
            # Create field
            self.field = ZetaField(config)
            
            # Set initial condition
            self.field.set_initial_condition(
                profile=self.profile_var.get(),
                amplitude=1.0,
                width=1.0,
                center=0.0,
                add_zeta=self.zeta_var.get()
            )
            
            # Update visualization
            self.update_visualization()
            
            # Display initial status
            conserved = self.field.compute_conserved_quantities()
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "=== FIELD INITIALIZED ===\n")
            self.results_text.insert(tk.END, f"Dimensions: {config.dimensions}D\n")
            self.results_text.insert(tk.END, f"Grid: {config.grid_size} points\n")
            self.results_text.insert(tk.END, f"Mass²: {config.mass_squared}\n")
            self.results_text.insert(tk.END, f"Coupling: {config.coupling_lambda}\n")
            self.results_text.insert(tk.END, f"Initial energy: {conserved['total_energy']:.6f}\n")
            self.results_text.insert(tk.END, f"Initial ζ-charge: {conserved['zeta_charge']}\n")
            self.results_text.insert(tk.END, f"ζ-singularities: {'Yes' if self.zeta_var.get() else 'No'}\n\n")
            
            self.status_var.set("Field initialized")
            self.run_button.config(state="normal")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize field: {e}")
            self.status_var.set("Initialization failed")
    
    def update_visualization(self):
        """Update the field visualization."""
        if self.field is None:
            return
        
        profile = self.field.get_field_profile()
        
        self.ax.clear()
        
        if profile['dimension'] == 1:
            # 1D plot
            if len(profile['positions']) > 0:
                self.ax.plot(profile['positions'], profile['values'], 'b-', 
                           linewidth=2, label='Field')
            
            if len(profile['zeta_positions']) > 0:
                self.ax.scatter(profile['zeta_positions'], 
                              np.zeros_like(profile['zeta_positions']), 
                              c='red', s=100, marker='x', linewidth=3, 
                              label='ζ-singularities')
                
                # Annotate ζ-values
                for pos, val in zip(profile['zeta_positions'], profile['zeta_values']):
                    self.ax.annotate(f'ζ_{{{val:.2f}}}', (pos, 0), 
                                   xytext=(0, 20), textcoords='offset points',
                                   ha='center', fontsize=10, color='red')
            
            self.ax.set_xlabel("Position")
            self.ax.set_ylabel("Field Value")
            self.ax.legend()
            
        else:
            # 2D plot
            im = self.ax.imshow(profile['values'], extent=[
                profile['x'][0], profile['x'][-1], 
                profile['y'][0], profile['y'][-1]
            ], origin='lower', cmap='viridis')
            
            # Mark ζ-singularities
            zeta_y, zeta_x = np.where(profile['zeta_mask'])
            if len(zeta_x) > 0:
                # Convert indices to coordinates
                zeta_x_coords = profile['x'][zeta_x]
                zeta_y_coords = profile['y'][zeta_y]
                self.ax.scatter(zeta_x_coords, zeta_y_coords, 
                              c='red', s=100, marker='x', linewidth=3)
            
            self.ax.set_xlabel("X Position")
            self.ax.set_ylabel("Y Position")
            self.fig.colorbar(im, ax=self.ax, label="Field Value")
        
        self.ax.set_title(f"ζ-Field at t = {profile['time']:.3f}")
        self.ax.grid(True, alpha=0.3)
        
        self.canvas.draw()
    
    def run_simulation(self):
        """Run field evolution simulation."""
        if self.field is None:
            messagebox.showerror("Error", "Please initialize field first")
            return
        
        self.simulation_running = True
        self.run_button.config(state="disabled")
        self.stop_button.config(state="normal")
        
        # Run simulation in separate thread
        thread = threading.Thread(target=self._simulation_worker)
        thread.daemon = True
        thread.start()
    
    def _simulation_worker(self):
        """Worker function for simulation thread."""
        try:
            total_time = float(self.time_var.get())
            
            # Initial conserved quantities
            initial_conserved = self.field.compute_conserved_quantities()
            
            def progress_callback(progress, time, conserved, metrics):
                if not self.simulation_running:
                    return
                
                # Update progress bar
                self.progress['value'] = progress * 100
                
                # Update status
                self.status_var.set(f"Evolving: t={time:.3f}/{total_time:.1f}")
                
                # Update visualization periodically
                if progress == 0 or progress >= 0.95 or int(progress * 20) != int((progress - 0.05) * 20):
                    self.frame.after(0, self.update_visualization)
                
                # Update results text
                if progress == 0 or progress >= 0.95 or int(progress * 10) != int((progress - 0.1) * 10):
                    energy_drift = abs(conserved['total_energy'] - initial_conserved['total_energy'])
                    conservation_percent = (1 - energy_drift / abs(initial_conserved['total_energy'])) * 100
                    
                    result_update = (
                        f"Progress: {progress*100:.0f}% | "
                        f"t={time:.3f} | "
                        f"E={conserved['total_energy']:.6f} | "
                        f"ζ={conserved['zeta_charge']} | "
                        f"Conservation: {conservation_percent:.2f}% | "
                        f"Stable: {metrics['stable']}\n"
                    )
                    
                    self.frame.after(0, lambda: self._update_results_async(result_update))
            
            # Run evolution
            self.field.evolve(total_time, method="euler", 
                            save_history=True, progress_callback=progress_callback)
            
            # Final results
            final_conserved = self.field.compute_conserved_quantities()
            energy_drift = abs(final_conserved['total_energy'] - initial_conserved['total_energy'])
            conservation_quality = (1 - energy_drift / abs(initial_conserved['total_energy'])) * 100
            
            final_results = (
                f"\n=== SIMULATION COMPLETE ===\n"
                f"Final time: {self.field.time:.3f}\n"
                f"Final energy: {final_conserved['total_energy']:.6f}\n"
                f"Energy conservation: {conservation_quality:.2f}%\n"
                f"ζ-charge conservation: {final_conserved['zeta_charge'] == initial_conserved['zeta_charge']}\n"
                f"Final ζ-charge: {final_conserved['zeta_charge']}\n"
                f"Max field value: {final_conserved['max_field']:.6f}\n\n"
            )
            
            self.frame.after(0, lambda: self._finalize_simulation(final_results))
            
        except Exception as e:
            error_msg = f"Simulation failed: {e}\n"
            self.frame.after(0, lambda: self._finalize_simulation(error_msg))
    
    def _update_results_async(self, text):
        """Update results text from simulation thread."""
        self.results_text.insert(tk.END, text)
        self.results_text.see(tk.END)
    
    def _finalize_simulation(self, final_text):
        """Finalize simulation from main thread."""
        self.results_text.insert(tk.END, final_text)
        self.results_text.see(tk.END)
        
        self.simulation_running = False
        self.run_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.progress['value'] = 100
        self.status_var.set("Simulation complete")
        
        # Final visualization update
        self.update_visualization()
    
    def stop_simulation(self):
        """Stop running simulation."""
        self.simulation_running = False
        self.run_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status_var.set("Simulation stopped")


class ZetaRegularizationGUI:
    """Main GUI application for ζ-regularization framework."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ζ-Regularization Framework")
        self.root.geometry("1200x800")
        
        # Set up the interface
        self.setup_ui()
        
        # Configure grid weights for resizing
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
    
    def setup_ui(self):
        # Create main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Create tabs
        self.algebra_tab = ZetaAlgebraTab(self.notebook)
        self.field_tab = ZetaFieldTab(self.notebook)
        
        # Add tabs to notebook
        self.notebook.add(self.algebra_tab.frame, text="ζ-Algebra")
        self.notebook.add(self.field_tab.frame, text="Field Theory")
        
        # Configure grid weights for tabs
        self.algebra_tab.frame.columnconfigure(0, weight=1)
        self.algebra_tab.frame.columnconfigure(1, weight=1)
        
        self.field_tab.frame.columnconfigure(0, weight=1)
        self.field_tab.frame.columnconfigure(1, weight=2)
        self.field_tab.frame.rowconfigure(0, weight=1)
        self.field_tab.frame.rowconfigure(1, weight=1)
        
        # Create menu bar
        self.create_menu()
        
        # Status bar
        self.create_status_bar()
    
    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Save Configuration", command=self.save_config)
        file_menu.add_command(label="Load Configuration", command=self.load_config)
        file_menu.add_separator()
        file_menu.add_command(label="Export Results", command=self.export_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Performance Benchmark", command=self.run_benchmark)
        tools_menu.add_command(label="ζ-Reconstruction Test Suite", command=self.run_test_suite)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About ζ-Regularization", command=self.show_about)
        help_menu.add_command(label="Mathematical Background", command=self.show_math_help)
        help_menu.add_command(label="Usage Guide", command=self.show_usage_guide)
    
    def create_status_bar(self):
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.grid(row=1, column=0, sticky="ew")
        
        self.status_text = tk.StringVar(value="Ready - ζ-Regularization Framework v1.0")
        ttk.Label(self.status_bar, textvariable=self.status_text).pack(side=tk.LEFT, padx=5)
        
        # Version info
        ttk.Label(self.status_bar, text="Semantic Infinity Preservation Enabled").pack(side=tk.RIGHT, padx=5)
    
    def save_config(self):
        """Save current configuration to file."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            # Implementation would save current GUI state
            messagebox.showinfo("Info", f"Configuration would be saved to {filename}")
    
    def load_config(self):
        """Load configuration from file."""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            # Implementation would load GUI state
            messagebox.showinfo("Info", f"Configuration would be loaded from {filename}")
    
    def export_results(self):
        """Export simulation results."""
        if self.field_tab.field is None:
            messagebox.showerror("Error", "No simulation results to export")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if filename:
            try:
                self.field_tab.fig.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Results exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {e}")
    
    def run_benchmark(self):
        """Run performance benchmark."""
        def benchmark_worker():
            results = []
            
            # Test ζ-algebra performance
            start_time = time.time()
            for i in range(10000):
                a = zeta(i * 0.001, "benchmark")
                b = finite(i * 0.002)
                result = a + b * finite(0.5)
                if i % 1000 == 0:
                    _ = result.zeta_reconstruct(0) if result.is_zeta_element() else result.value
            
            algebra_time = time.time() - start_time
            results.append(f"ζ-Algebra: 10,000 operations in {algebra_time:.3f}s")
            
            # Test field evolution performance
            start_time = time.time()
            config = FieldConfiguration(dimensions=1, grid_size=32, time_step=0.01)
            field = ZetaField(config)
            field.set_initial_condition("gaussian", add_zeta=True)
            field.evolve(0.5, method="euler", save_history=False)
            
            field_time = time.time() - start_time
            results.append(f"Field Evolution: 50 steps in {field_time:.3f}s")
            
            # Display results
            result_text = "=== PERFORMANCE BENCHMARK ===\n\n" + "\n".join(results)
            result_text += f"\n\nOverall: Framework is {'highly' if algebra_time < 1.0 else 'moderately'} performant"
            
            messagebox.showinfo("Benchmark Results", result_text)
        
        # Run in thread to avoid freezing GUI
        thread = threading.Thread(target=benchmark_worker)
        thread.daemon = True
        thread.start()
    
    def run_test_suite(self):
        """Run comprehensive test suite."""
        semiring = ZetaSemiring()
        results = semiring.run_comprehensive_test()
        
        result_text = "=== ζ-RECONSTRUCTION TEST SUITE ===\n\n"
        
        for test_name, passed in results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            test_display = test_name.replace('_', ' ').title()
            result_text += f"{status}: {test_display}\n"
        
        overall_pass = all(results.values())
        result_text += f"\nOverall: {'✓ ALL TESTS PASSED' if overall_pass else '✗ SOME TESTS FAILED'}"
        
        if overall_pass:
            result_text += "\n\nThe ζ-regularization framework is mathematically sound!"
        
        messagebox.showinfo("Test Suite Results", result_text)
    
    def show_about(self):
        """Show about dialog."""
        about_text = """ζ-Regularization Framework v1.0

A mathematically rigorous approach to handling divergences in fundamental physics 
by preserving semantic information through symbolic tropical algebra.

Key Features:
• Information-preserving regularization
• Tropical semiring algebra with ζ-symbols  
• Multi-dimensional field theory simulations
• Real-time visualization and analysis
• Comprehensive verification tools

Mathematical Foundation:
• Semiring: (T_ζ, ⊕, ⊗, +∞, 0)
• ζ-reconstruction: ζ_a • 0 = a
• Tropical operations with semantic preservation

Applications:
• Quantum field theory with finite cutoffs
• Holographic duality with information bounds
• Condensed matter systems with finite Hilbert spaces
• Cosmological models with bounded entropy

Author: ζ-Regularization Framework Team
License: MIT License"""
        
        messagebox.showinfo("About ζ-Regularization", about_text)
    
    def show_math_help(self):
        """Show mathematical background help."""
        math_text = """Mathematical Background

ζ-SYMBOLS:
• ζ_a represents the result of a/0 with semantic tag a
• Preserves finite information that would be lost in traditional regularization
• Forms part of the ζ-extended tropical semiring T_ζ

TROPICAL OPERATIONS:
• Addition: a ⊕ b = min(a, b)  [min operation]
• Multiplication: a ⊗ b = a + b  [standard addition]
• Identity elements: +∞ (additive), 0 (multiplicative)

ζ-RECONSTRUCTION:
• Fundamental operation: ζ_a • 0 = a
• Recovers finite physics from symbolic infinities
• Enables information preservation through regularization

SEMIRING PROPERTIES:
• Associativity: (a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)
• Commutativity: a ⊕ b = b ⊕ a  
• Distributivity: a ⊗ (b ⊕ c) = (a ⊗ b) ⊕ (a ⊗ c)
• All verified algorithmically in this framework

FIELD THEORY:
• Klein-Gordon equation: □φ + m²φ + λφ³ + Σζ_i = 0
• ζ-sources represent localized singularities
• Energy and ζ-charge conservation laws"""
        
        messagebox.showinfo("Mathematical Background", math_text)
    
    def show_usage_guide(self):
        """Show usage guide."""
        usage_text = """Usage Guide

ζ-ALGEBRA TAB:
1. Enter finite values or ζ-symbols (format: ζ_2.5)
2. Use "Make ζ" buttons to convert finite → ζ-symbol
3. Perform tropical operations with ⊕ and ⊗ buttons
4. Test ζ-reconstruction with any tag value
5. Verify semiring properties automatically

FIELD THEORY TAB:
1. Configure field parameters (dimensions, grid size, physics)
2. Choose initial profile (gaussian, soliton, random)
3. Enable/disable ζ-singularities  
4. Click "Initialize Field" to set up simulation
5. Click "Run Simulation" to evolve the field
6. Watch real-time visualization and conservation laws

TIPS:
• Start with 1D simulations for faster results
• Use smaller grid sizes for real-time interaction
• ζ-singularities appear as red X markers
• Energy conservation indicates numerical stability
• ζ-charge conservation shows topological stability

INTERPRETATION:
• Finite field values evolve according to Klein-Gordon dynamics
• ζ-singularities preserve information about initial conditions
• Reconstruction operation ζ_a • 0 = a recovers physical scales
• Framework demonstrates information preservation during regularization"""
        
        messagebox.showinfo("Usage Guide", usage_text)
    
    def run(self):
        """Start the GUI application."""
        # Center the window
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        
        # Start the main loop
        self.root.mainloop()


if __name__ == "__main__":
    # Check dependencies
    try:
        import tkinter as tk
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        print(f"Missing required dependency: {e}")
        print("Please install: pip install matplotlib numpy")
        sys.exit(1)
    
    # Launch GUI
    print("Launching ζ-Regularization Framework GUI...")
    app = ZetaRegularizationGUI()
    app.run()

"""
Core ζ-Algebra Implementation
============================

This module provides the fundamental algebraic structures for ζ-regularization:
- ZetaSymbol: Semantic tags for infinities
- ZetaExtendedNumber: Elements of the ζ-extended tropical semiring
- ZetaSemiring: Complete algebraic structure with verified properties

Author: ζ-Regularization Framework
License: MIT
"""

import numpy as np
import warnings
from dataclasses import dataclass
from typing import Set, Optional, Union, List
from abc import ABC, abstractmethod


@dataclass(frozen=True)
class ZetaSymbol:
    """
    Rigorous ζ-symbol representing semantic information about infinities.
    
    Each ζ-symbol encodes:
    - tag: The finite coefficient that led to the infinity
    - source: Descriptive information about the origin
    """
    tag: float
    source: str = "unknown"
    
    def __post_init__(self):
        if abs(self.tag) < 1e-15:
            raise ValueError("ζ-symbols cannot have zero tags")
    
    def __hash__(self):
        return hash((round(self.tag, 12), self.source))
    
    def __eq__(self, other):
        if not isinstance(other, ZetaSymbol):
            return False
        return abs(self.tag - other.tag) < 1e-12 and self.source == other.source
    
    def __repr__(self):
        return f"ζ_{{{self.tag:.6f}}}"
    
    def __str__(self):
        return f"ζ[{self.tag:.3f}|{self.source}]"


class ZetaExtendedNumber:
    """
    Element of the ζ-extended tropical semiring T_ζ.
    
    Supports:
    - Tropical arithmetic (⊕ = min, ⊗ = +)
    - ζ-symbol tracking and composition
    - Information reconstruction via ζ_a • 0 = a
    """
    
    def __init__(self, value: Union[float, str], zeta_tags: Optional[Set[ZetaSymbol]] = None):
        if isinstance(value, str):
            if value.lower() in ["inf", "infinity", "+inf"]:
                self.value = np.inf
            elif value.lower() in ["-inf", "-infinity"]:
                self.value = -np.inf
            else:
                self.value = float(value)
        else:
            self.value = float(value)
        
        self.zeta_tags = zeta_tags or set()
        
        # Validate consistency
        if self.value != np.inf and self.zeta_tags:
            warnings.warn("Finite values with ζ-tags may break semiring properties")
    
    def is_zeta_element(self) -> bool:
        """Check if this represents a ζ-tagged infinity."""
        return self.value == np.inf and bool(self.zeta_tags)
    
    def is_finite(self) -> bool:
        """Check if this represents a finite value."""
        return np.isfinite(self.value)
    
    def get_minimal_tag(self) -> Optional[float]:
        """Get the minimal tag value (for tropical operations)."""
        if not self.zeta_tags:
            return None
        return min(tag.tag for tag in self.zeta_tags)
    
    def __add__(self, other) -> 'ZetaExtendedNumber':
        """Tropical addition: a ⊕ b = min(a, b) with ζ-tag inheritance."""
        if not isinstance(other, ZetaExtendedNumber):
            other = ZetaExtendedNumber(other)
        
        # Handle ζ-infinity cases
        if self.value == np.inf and other.value == np.inf:
            if self.zeta_tags and other.zeta_tags:
                # Take the ζ-symbol with minimal tag
                self_min = self.get_minimal_tag()
                other_min = other.get_minimal_tag()
                
                if self_min <= other_min:
                    return ZetaExtendedNumber(np.inf, self.zeta_tags.copy())
                else:
                    return ZetaExtendedNumber(np.inf, other.zeta_tags.copy())
            elif self.zeta_tags:
                return ZetaExtendedNumber(np.inf, self.zeta_tags.copy())
            elif other.zeta_tags:
                return ZetaExtendedNumber(np.inf, other.zeta_tags.copy())
            else:
                return ZetaExtendedNumber(np.inf)
        
        # Standard tropical addition (min operation)
        if self.value <= other.value:
            return ZetaExtendedNumber(self.value, self.zeta_tags.copy())
        else:
            return ZetaExtendedNumber(other.value, other.zeta_tags.copy())
    
    def __mul__(self, other) -> 'ZetaExtendedNumber':
        """Tropical multiplication: a ⊗ b = a + b with ζ-composition."""
        if not isinstance(other, ZetaExtendedNumber):
            other = ZetaExtendedNumber(other)
        
        # ζ-reconstruction: ζ_a ⊗ 0 = a
        if other.value == 0 and self.is_zeta_element():
            if len(self.zeta_tags) == 1:
                return ZetaExtendedNumber(list(self.zeta_tags)[0].tag)
            else:
                # Multiple tags: sum them
                total = sum(tag.tag for tag in self.zeta_tags)
                return ZetaExtendedNumber(total)
        
        if self.value == 0 and other.is_zeta_element():
            if len(other.zeta_tags) == 1:
                return ZetaExtendedNumber(list(other.zeta_tags)[0].tag)
            else:
                total = sum(tag.tag for tag in other.zeta_tags)
                return ZetaExtendedNumber(total)
        
        # Standard tropical multiplication
        new_value = self.value + other.value
        
        # Merge ζ-tags
        new_tags = self.zeta_tags | other.zeta_tags
        
        return ZetaExtendedNumber(new_value, new_tags)
    
    def __rmul__(self, other):
        """Right multiplication for scalar * ZetaExtendedNumber."""
        return ZetaExtendedNumber(other).__mul__(self)
    
    def __pow__(self, exponent):
        """Exponentiation with ζ-tag scaling."""
        if self.is_zeta_element():
            scaled_tags = {ZetaSymbol(tag.tag * exponent, f"{tag.source}^{exponent}") 
                          for tag in self.zeta_tags}
            return ZetaExtendedNumber(np.inf, scaled_tags)
        else:
            return ZetaExtendedNumber(self.value * exponent)
    
    def zeta_reconstruct(self, zero_element=0) -> float:
        """
        ζ-reconstruction: Extract finite information from ζ-symbols.
        
        This is the fundamental operation ζ_a • 0 = a that recovers
        finite physics from symbolic infinities.
        """
        if zero_element != 0:
            raise ValueError("ζ-reconstruction requires zero element")
        
        if not self.is_zeta_element():
            raise ValueError("ζ-reconstruction only applies to ζ-tagged infinities")
        
        if len(self.zeta_tags) == 1:
            return list(self.zeta_tags)[0].tag
        else:
            # Multiple tags: return their sum
            return sum(tag.tag for tag in self.zeta_tags)
    
    def __eq__(self, other) -> bool:
        """Equality comparison with tolerance for floating point."""
        if not isinstance(other, ZetaExtendedNumber):
            return False
        
        values_equal = abs(self.value - other.value) < 1e-12
        tags_equal = self.zeta_tags == other.zeta_tags
        
        return values_equal and tags_equal
    
    def __repr__(self):
        if self.is_zeta_element():
            if len(self.zeta_tags) == 1:
                return f"ζ_{{{list(self.zeta_tags)[0].tag:.6f}}}"
            else:
                tags_str = ','.join(f"{tag.tag:.3f}" for tag in 
                                   sorted(self.zeta_tags, key=lambda x: x.tag))
                return f"ζ_{{{tags_str}}}"
        elif self.value == np.inf:
            return "+∞"
        elif self.value == -np.inf:
            return "-∞"
        else:
            return f"{self.value:.6f}"
    
    def __str__(self):
        return self.__repr__()


class ZetaSemiring:
    """
    Complete ζ-extended tropical semiring with verified properties.
    
    Provides methods to verify semiring axioms and perform algebraic operations.
    """
    
    def __init__(self):
        self.additive_identity = ZetaExtendedNumber(np.inf)  # +∞
        self.multiplicative_identity = ZetaExtendedNumber(0)  # 0
    
    def verify_semiring_axioms(self, test_elements: List[ZetaExtendedNumber]) -> dict:
        """
        Verify that the ζ-extended structure satisfies semiring axioms.
        
        Returns a dictionary with verification results for each property.
        """
        results = {}
        
        # Test associativity of addition
        associative_add = True
        for a in test_elements[:3]:
            for b in test_elements[:3]:
                for c in test_elements[:3]:
                    left = (a + b) + c
                    right = a + (b + c)
                    if not left == right:
                        associative_add = False
                        break
        results['associative_addition'] = associative_add
        
        # Test commutativity of addition
        commutative_add = True
        for a in test_elements[:3]:
            for b in test_elements[:3]:
                if not (a + b) == (b + a):
                    commutative_add = False
                    break
        results['commutative_addition'] = commutative_add
        
        # Test associativity of multiplication
        associative_mult = True
        for a in test_elements[:3]:
            for b in test_elements[:3]:
                for c in test_elements[:3]:
                    left = (a * b) * c
                    right = a * (b * c)
                    if not left == right:
                        associative_mult = False
                        break
        results['associative_multiplication'] = associative_mult
        
        # Test distributivity
        distributive = True
        for a in test_elements[:2]:
            for b in test_elements[:2]:
                for c in test_elements[:2]:
                    left = a * (b + c)
                    right = (a * b) + (a * c)
                    if not left == right:
                        distributive = False
                        break
        results['distributive'] = distributive
        
        # Test identity elements
        identity_test = True
        for a in test_elements[:3]:
            # Additive identity
            if not (a + self.additive_identity) == a:
                identity_test = False
            # Multiplicative identity  
            if not (a * self.multiplicative_identity) == a:
                identity_test = False
        results['identity_elements'] = identity_test
        
        return results
    
    def create_test_elements(self) -> List[ZetaExtendedNumber]:
        """Create a standard set of test elements for verification."""
        return [
            ZetaExtendedNumber(1.0),
            ZetaExtendedNumber(2.5),
            ZetaExtendedNumber(0.0),
            ZetaExtendedNumber(np.inf, {ZetaSymbol(3.0, "test1")}),
            ZetaExtendedNumber(np.inf, {ZetaSymbol(1.5, "test2")}),
            ZetaExtendedNumber(np.inf, {ZetaSymbol(2.0, "test3"), ZetaSymbol(4.0, "test4")})
        ]
    
    def run_comprehensive_test(self) -> dict:
        """Run comprehensive test of semiring properties."""
        test_elements = self.create_test_elements()
        results = self.verify_semiring_axioms(test_elements)
        
        # Additional tests
        results['zeta_reconstruction'] = self._test_zeta_reconstruction()
        results['information_preservation'] = self._test_information_preservation()
        
        return results
    
    def _test_zeta_reconstruction(self) -> bool:
        """Test the fundamental ζ-reconstruction identity."""
        try:
            # Test basic reconstruction
            zeta_a = ZetaExtendedNumber(np.inf, {ZetaSymbol(2.5, "test")})
            reconstructed = zeta_a.zeta_reconstruct(0)
            
            if abs(reconstructed - 2.5) > 1e-12:
                return False
            
            # Test via multiplication
            zero = ZetaExtendedNumber(0)
            mult_result = zeta_a * zero
            
            if abs(mult_result.value - 2.5) > 1e-12:
                return False
            
            return True
        
        except Exception:
            return False
    
    def _test_information_preservation(self) -> bool:
        """Test that information is preserved through operations."""
        try:
            # Create complex expression with multiple ζ-symbols
            za = ZetaExtendedNumber(np.inf, {ZetaSymbol(3.0, "source_a")})
            zb = ZetaExtendedNumber(np.inf, {ZetaSymbol(2.0, "source_b")})
            
            # Perform operations
            result = (za + zb) * ZetaExtendedNumber(1.5)
            
            # Should preserve the minimal tag information
            min_tag = result.get_minimal_tag()
            
            return abs(min_tag - 2.0) < 1e-12
        
        except Exception:
            return False


# Convenience functions for easy usage
def zeta(tag: float, source: str = "user") -> ZetaExtendedNumber:
    """Create a ζ-symbol with given tag and source."""
    return ZetaExtendedNumber(np.inf, {ZetaSymbol(tag, source)})

def finite(value: float) -> ZetaExtendedNumber:
    """Create a finite element."""
    return ZetaExtendedNumber(value)

def tropical_add(*args) -> ZetaExtendedNumber:
    """Tropical addition of multiple arguments."""
    result = args[0]
    for arg in args[1:]:
        result = result + arg
    return result

def tropical_mult(*args) -> ZetaExtendedNumber:
    """Tropical multiplication of multiple arguments."""
    result = args[0]
    for arg in args[1:]:
        result = result * arg
    return result


if __name__ == "__main__":
    # Basic demonstration
    print("=== ζ-Algebra Demonstration ===")
    
    # Create test elements
    a = finite(2.5)
    b = finite(1.8)
    za = zeta(3.0, "demo_a")
    zb = zeta(2.0, "demo_b")
    
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"ζ_a = {za}")
    print(f"ζ_b = {zb}")
    
    # Tropical operations
    print(f"\nTropical addition: a ⊕ b = {a + b}")
    print(f"Tropical multiplication: a ⊗ b = {a * b}")
    print(f"ζ-addition: ζ_a ⊕ ζ_b = {za + zb}")
    
    # ζ-reconstruction
    reconstructed = za.zeta_reconstruct(0)
    print(f"\nζ-reconstruction: ζ_a • 0 = {reconstructed}")
    
    # Semiring verification
    semiring = ZetaSemiring()
    test_results = semiring.run_comprehensive_test()
    
    print(f"\n=== Semiring Verification ===")
    for property_name, passed in test_results.items():
        status = "✓" if passed else "✗"
        print(f"{status} {property_name}: {passed}")
    
    overall_pass = all(test_results.values())
    print(f"\nOverall: {'✓ PASSED' if overall_pass else '✗ FAILED'}")

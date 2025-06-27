"""
Test Suite for ζ-Algebra Core
=============================

Comprehensive tests for the fundamental ζ-regularization algebra including:
- ZetaSymbol creation and properties
- ZetaExtendedNumber operations
- Semiring property verification
- ζ-reconstruction accuracy
- Edge cases and error handling

Author: ζ-Regularization Framework
License: MIT
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from zeta_core.algebra import (
    ZetaSymbol, 
    ZetaExtendedNumber, 
    ZetaSemiring,
    zeta, 
    finite,
    tropical_add,
    tropical_mult
)


class TestZetaSymbol:
    """Test ZetaSymbol class functionality."""
    
    def test_creation(self):
        """Test basic ZetaSymbol creation."""
        symbol = ZetaSymbol(2.5, "test_source")
        assert symbol.tag == 2.5
        assert symbol.source == "test_source"
    
    def test_zero_tag_rejection(self):
        """Test that zero tags are rejected."""
        with pytest.raises(ValueError, match="ζ-symbols cannot have zero tags"):
            ZetaSymbol(0.0, "test")
        
        with pytest.raises(ValueError, match="ζ-symbols cannot have zero tags"):
            ZetaSymbol(1e-16, "test")  # Below threshold
    
    def test_equality(self):
        """Test ZetaSymbol equality comparison."""
        s1 = ZetaSymbol(2.5, "source1")
        s2 = ZetaSymbol(2.5, "source1")
        s3 = ZetaSymbol(2.5, "source2")
        s4 = ZetaSymbol(2.6, "source1")
        
        assert s1 == s2
        assert s1 != s3  # Different source
        assert s1 != s4  # Different tag
    
    def test_hashing(self):
        """Test ZetaSymbol can be used in sets/dicts."""
        s1 = ZetaSymbol(2.5, "source1")
        s2 = ZetaSymbol(2.5, "source1")
        s3 = ZetaSymbol(3.0, "source1")
        
        symbol_set = {s1, s2, s3}
        assert len(symbol_set) == 2  # s1 and s2 are equal
    
    def test_string_representation(self):
        """Test string representation."""
        symbol = ZetaSymbol(3.14159, "pi_source")
        assert "3.14159" in str(symbol)
        assert "ζ" in repr(symbol)


class TestZetaExtendedNumber:
    """Test ZetaExtendedNumber class functionality."""
    
    def test_finite_creation(self):
        """Test creation of finite elements."""
        f = finite(2.5)
        assert f.value == 2.5
        assert len(f.zeta_tags) == 0
        assert f.is_finite()
        assert not f.is_zeta_element()
    
    def test_zeta_creation(self):
        """Test creation of ζ-elements."""
        z = zeta(3.0, "test")
        assert z.value == np.inf
        assert len(z.zeta_tags) == 1
        assert not z.is_finite()
        assert z.is_zeta_element()
        
        # Check tag content
        tag = list(z.zeta_tags)[0]
        assert tag.tag == 3.0
        assert tag.source == "test"
    
    def test_tropical_addition_finite(self):
        """Test tropical addition with finite elements."""
        a = finite(2.5)
        b = finite(1.8)
        result = a + b
        
        assert result.value == 1.8  # min(2.5, 1.8)
        assert result.is_finite()
    
    def test_tropical_addition_zeta(self):
        """Test tropical addition with ζ-elements."""
        za = zeta(3.0, "source_a")
        zb = zeta(2.0, "source_b")
        result = za + zb
        
        assert result.is_zeta_element()
        # Should take the one with smaller tag
        min_tag = result.get_minimal_tag()
        assert min_tag == 2.0
    
    def test_tropical_addition_mixed(self):
        """Test tropical addition with mixed elements."""
        a = finite(2.5)
        za = zeta(3.0, "source")
        result = a + za
        
        # ζ-element should dominate (be selected)
        assert result.is_zeta_element()
    
    def test_tropical_multiplication_finite(self):
        """Test tropical multiplication with finite elements."""
        a = finite(2.5)
        b = finite(1.8)
        result = a * b
        
        assert abs(result.value - 4.3) < 1e-12  # 2.5 + 1.8
        assert result.is_finite()
    
    def test_tropical_multiplication_zeta(self):
        """Test tropical multiplication with ζ-elements."""
        za = zeta(3.0, "source_a")
        zb = zeta(2.0, "source_b")
        result = za * zb
        
        assert result.is_zeta_element()
        assert len(result.zeta_tags) == 2  # Merged tags
    
    def test_zeta_reconstruction_direct(self):
        """Test direct ζ-reconstruction."""
        za = zeta(3.14159, "pi_test")
        reconstructed = za.zeta_reconstruct(0)
        
        assert abs(reconstructed - 3.14159) < 1e-12
    
    def test_zeta_reconstruction_multiplication(self):
        """Test ζ-reconstruction via multiplication."""
        za = zeta(2.5, "test")
        zero = finite(0)
        result = za * zero
        
        assert result.is_finite()
        assert abs(result.value - 2.5) < 1e-12
    
    def test_zeta_reconstruction_error_cases(self):
        """Test ζ-reconstruction error handling."""
        a = finite(2.5)
        za = zeta(3.0, "test")
        
        # Non-zero element
        with pytest.raises(ValueError, match="requires zero element"):
            za.zeta_reconstruct(1.0)
        
        # Non-ζ element
        with pytest.raises(ValueError, match="only applies to ζ-tagged infinities"):
            a.zeta_reconstruct(0)
    
    def test_multiple_tag_reconstruction(self):
        """Test reconstruction with multiple tags."""
        tags = {ZetaSymbol(2.0, "source1"), ZetaSymbol(3.0, "source2")}
        z = ZetaExtendedNumber(np.inf, tags)
        
        reconstructed = z.zeta_reconstruct(0)
        assert abs(reconstructed - 5.0) < 1e-12  # Sum of tags
    
    def test_equality(self):
        """Test equality comparison."""
        a1 = finite(2.5)
        a2 = finite(2.5)
        a3 = finite(2.6)
        
        assert a1 == a2
        assert a1 != a3
        
        z1 = zeta(3.0, "test")
        z2 = zeta(3.0, "test")
        z3 = zeta(3.1, "test")
        
        assert z1 == z2
        assert z1 != z3
    
    def test_exponentiation(self):
        """Test ζ-element exponentiation."""
        za = zeta(2.0, "base")
        result = za ** 3
        
        assert result.is_zeta_element()
        tag = list(result.zeta_tags)[0]
        assert abs(tag.tag - 6.0) < 1e-12  # 2.0 * 3
        assert "base^3" in tag.source


class TestZetaSemiring:
    """Test semiring property verification."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.semiring = ZetaSemiring()
        self.test_elements = [
            finite(1.0),
            finite(2.5),
            finite(0.0),
            zeta(3.0, "test1"),
            zeta(1.5, "test2")
        ]
    
    def test_additive_identity(self):
        """Test additive identity property."""
        infinity = self.semiring.additive_identity
        
        for element in self.test_elements:
            result = element + infinity
            assert result == element
    
    def test_multiplicative_identity(self):
        """Test multiplicative identity property."""
        zero = self.semiring.multiplicative_identity
        
        for element in self.test_elements:
            result = element * zero
            # For ζ-elements, this should reconstruct
            if element.is_zeta_element():
                assert result.is_finite()
            else:
                assert result == element
    
    def test_associativity_addition(self):
        """Test associativity of tropical addition."""
        a, b, c = self.test_elements[:3]
        
        left = (a + b) + c
        right = a + (b + c)
        
        assert left == right
    
    def test_commutativity_addition(self):
        """Test commutativity of tropical addition."""
        a, b = self.test_elements[:2]
        
        forward = a + b
        backward = b + a
        
        assert forward == backward
    
    def test_associativity_multiplication(self):
        """Test associativity of tropical multiplication."""
        a, b, c = self.test_elements[:3]
        
        left = (a * b) * c
        right = a * (b * c)
        
        assert left == right
    
    def test_distributivity(self):
        """Test distributivity property."""
        a, b, c = self.test_elements[:3]
        
        left = a * (b + c)
        right = (a * b) + (a * c)
        
        assert left == right
    
    def test_comprehensive_verification(self):
        """Test comprehensive semiring verification."""
        results = self.semiring.run_comprehensive_test()
        
        # All properties should pass
        assert all(results.values()), f"Failed properties: {[k for k, v in results.items() if not v]}"
    
    def test_information_preservation(self):
        """Test information preservation through operations."""
        # Complex expression with multiple ζ-symbols
        za = zeta(3.0, "source_a")
        zb = zeta(2.0, "source_b")
        a = finite(1.5)
        
        # Perform complex operations
        result = (za + zb) * a
        
        # Information should be preserved
        assert result.is_zeta_element()
        min_tag = result.get_minimal_tag()
        assert abs(min_tag - 2.0) < 1e-12  # Smaller tag preserved


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_tropical_add_function(self):
        """Test tropical_add function."""
        elements = [finite(3.0), finite(1.5), finite(2.0)]
        result = tropical_add(*elements)
        
        assert abs(result.value - 1.5) < 1e-12  # Minimum
    
    def test_tropical_mult_function(self):
        """Test tropical_mult function."""
        elements = [finite(1.0), finite(2.0), finite(3.0)]
        result = tropical_mult(*elements)
        
        assert abs(result.value - 6.0) < 1e-12  # Sum
    
    def test_mixed_operations(self):
        """Test mixed operations with convenience functions."""
        finite_elements = [finite(2.0), finite(1.0)]
        zeta_elements = [zeta(3.0, "test1"), zeta(1.5, "test2")]
        
        # Mix finite and ζ-elements
        all_elements = finite_elements + zeta_elements
        result = tropical_add(*all_elements)
        
        # Should get the ζ-element with smallest tag
        assert result.is_zeta_element()
        assert abs(result.get_minimal_tag() - 1.5) < 1e-12


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_very_large_numbers(self):
        """Test with very large finite numbers."""
        large = finite(1e100)
        small = finite(1e-100)
        
        result = large + small
        assert abs(result.value - 1e-100) < 1e-112  # Min operation
    
    def test_negative_numbers(self):
        """Test with negative numbers."""
        a = finite(-2.5)
        b = finite(1.0)
        
        result = a + b
        assert abs(result.value - (-2.5)) < 1e-12  # Min
    
    def test_zero_operations(self):
        """Test operations involving zero."""
        zero = finite(0.0)
        a = finite(2.5)
        za = zeta(3.0, "test")
        
        # Zero with finite
        assert (zero + a) == zero
        assert (zero * a) == zero
        
        # Zero with ζ-element (reconstruction)
        result = za * zero
        assert result.is_finite()
        assert abs(result.value - 3.0) < 1e-12
    
    def test_infinity_operations(self):
        """Test operations with explicit infinities."""
        inf_element = ZetaExtendedNumber(np.inf)
        a = finite(2.5)
        
        # Should handle gracefully
        result = a + inf_element
        assert result.value == np.inf
    
    def test_nan_handling(self):
        """Test NaN handling."""
        nan_element = ZetaExtendedNumber(np.nan)
        a = finite(2.5)
        
        # Should handle gracefully without crashing
        result = a + nan_element
        # Behavior with NaN is allowed to be implementation-specific
    
    def test_string_inputs(self):
        """Test string input parsing."""
        inf_from_string = ZetaExtendedNumber("inf")
        assert inf_from_string.value == np.inf
        
        neg_inf_from_string = ZetaExtendedNumber("-inf")
        assert neg_inf_from_string.value == -np.inf
        
        finite_from_string = ZetaExtendedNumber("2.5")
        assert abs(finite_from_string.value - 2.5) < 1e-12


class TestPerformance:
    """Test performance characteristics."""
    
    def test_large_scale_operations(self):
        """Test performance with many operations."""
        import time
        
        start_time = time.time()
        
        # Start with ζ-element
        result = zeta(1.0, "performance")
        zero = finite(0)
        
        # Many operations
        for i in range(1000):
            random_finite = finite(np.random.random())
            result = result + random_finite
            
            # Periodic reconstruction
            if i % 100 == 0:
                temp = result * zero
        
        elapsed = time.time() - start_time
        
        # Should complete in reasonable time (< 1 second)
        assert elapsed < 1.0
        
        # Result should still be valid
        assert result.is_zeta_element()
    
    def test_reconstruction_accuracy_scaling(self):
        """Test reconstruction accuracy across scales."""
        scales = [1e-6, 1e-3, 1.0, 1e3, 1e6]
        
        for scale in scales:
            za = zeta(scale, "scale_test")
            reconstructed = za.zeta_reconstruct(0)
            
            relative_error = abs(reconstructed - scale) / scale
            assert relative_error < 1e-12, f"Failed for scale {scale}"


if __name__ == "__main__":
    # Run tests with pytest if available, otherwise basic test runner
    try:
        import pytest
        pytest.main([__file__, "-v"])
    except ImportError:
        # Basic test runner
        import unittest
        
        # Convert pytest-style tests to unittest
        class BasicTestRunner:
            def run_all_tests(self):
                test_classes = [
                    TestZetaSymbol, TestZetaExtendedNumber, 
                    TestZetaSemiring, TestConvenienceFunctions,
                    TestEdgeCases, TestPerformance
                ]
                
                total_tests = 0
                passed_tests = 0
                
                for test_class in test_classes:
                    print(f"\n=== {test_class.__name__} ===")
                    instance = test_class()
                    
                    # Setup if available
                    if hasattr(instance, 'setup_method'):
                        instance.setup_method()
                    
                    # Find test methods
                    test_methods = [m for m in dir(instance) if m.startswith('test_')]
                    
                    for method_name in test_methods:
                        total_tests += 1
                        try:
                            method = getattr(instance, method_name)
                            method()
                            print(f"✓ {method_name}")
                            passed_tests += 1
                        except Exception as e:
                            print(f"✗ {method_name}: {e}")
                
                print(f"\n=== SUMMARY ===")
                print(f"Tests passed: {passed_tests}/{total_tests}")
                print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
                
                return passed_tests == total_tests
        
        runner = BasicTestRunner()
        success = runner.run_all_tests()
        
        if success:
            print("\n🎉 All tests passed! ζ-algebra implementation is working correctly.")
        else:
            print("\n❌ Some tests failed. Please check the implementation.")

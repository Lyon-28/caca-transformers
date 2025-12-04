"""Tests for utility functions"""

import pytest
from caca_transformers import (
    CACA_VARIANTS,
    create_caca_model,
    estimate_training_tokens,
    estimate_batch_size,
    estimate_gpu_recommendation,
)


class TestUtilities:
    """Test suite for utility functions"""
    
    def test_caca_variants_exists(self):
        """Test that CACA_VARIANTS dictionary exists"""
        assert CACA_VARIANTS is not None
        assert len(CACA_VARIANTS) > 0
    
    def test_variant_structure(self):
        """Test variant structure"""
        for variant_name, variant_config in CACA_VARIANTS.items():
            assert 'vocab_size' in variant_config
            assert 'hidden_size' in variant_config
            assert 'num_hidden_layers' in variant_config
            assert 'num_attention_heads' in variant_config
            assert 'num_key_value_heads' in variant_config
    
    def test_create_small_model(self):
        """Test creating a small model"""
        model, config = create_caca_model("caca-10M")
        assert model is not None
        assert config is not None
        assert config.vocab_size > 0
    
    def test_create_invalid_variant(self):
        """Test creating invalid variant"""
        with pytest.raises(ValueError):
            create_caca_model("caca-invalid")
    
    def test_estimate_training_tokens(self):
        """Test training token estimation"""
        # Small model
        tokens_10m = estimate_training_tokens(10e6)
        assert isinstance(tokens_10m, str)
        assert 'M' in tokens_10m or 'B' in tokens_10m
        
        # Large model
        tokens_70b = estimate_training_tokens(70e9)
        assert isinstance(tokens_70b, str)
        assert 'B' in tokens_70b or 'T' in tokens_70b
    
    def test_estimate_batch_size(self):
        """Test batch size estimation"""
        # Small model
        batch_10m = estimate_batch_size(10e6)
        assert isinstance(batch_10m, str)
        
        # Large model
        batch_70b = estimate_batch_size(70e9)
        assert isinstance(batch_70b, str)
    
    def test_estimate_gpu_recommendation(self):
        """Test GPU recommendation"""
        # Small model
        gpu_10m = estimate_gpu_recommendation(10e6)
        assert isinstance(gpu_10m, str)
        assert len(gpu_10m) > 0
        
        # Large model
        gpu_70b = estimate_gpu_recommendation(70e9)
        assert isinstance(gpu_70b, str)
        assert 'A100' in gpu_70b or 'H100' in gpu_70b
    
    @pytest.mark.parametrize("variant_name", [
        "caca-10M",
        "caca-100M",
        "caca-1B",
    ])
    def test_multiple_variants(self, variant_name):
        """Test creating multiple variants"""
        model, config = create_caca_model(variant_name)
        assert model is not None
        assert config is not None
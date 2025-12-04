"""Tests for Caca configuration"""

import pytest
from caca_transformers import CacaConfig


class TestCacaConfiguration:
    """Test suite for Caca configuration"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = CacaConfig()
        assert config.vocab_size == 100000
        assert config.hidden_size == 2048
        assert config.num_hidden_layers == 24
        assert config.num_attention_heads == 16
        assert config.num_key_value_heads == 4
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = CacaConfig(
            vocab_size=50000,
            hidden_size=1024,
            num_hidden_layers=12,
            num_attention_heads=8,
            num_key_value_heads=2,
        )
        assert config.vocab_size == 50000
        assert config.hidden_size == 1024
        assert config.head_dim == 128
    
    def test_invalid_vocab_size(self):
        """Test invalid vocab size"""
        with pytest.raises(ValueError):
            CacaConfig(vocab_size=0)
        
        with pytest.raises(ValueError):
            CacaConfig(vocab_size=-100)
    
    def test_invalid_hidden_size(self):
        """Test invalid hidden size"""
        with pytest.raises(ValueError):
            CacaConfig(hidden_size=0)
    
    def test_invalid_head_configuration(self):
        """Test invalid attention head configuration"""
        # hidden_size not divisible by num_attention_heads
        with pytest.raises(ValueError):
            CacaConfig(hidden_size=1000, num_attention_heads=16)
        
        # num_attention_heads not divisible by num_key_value_heads
        with pytest.raises(ValueError):
            CacaConfig(num_attention_heads=16, num_key_value_heads=5)
    
    def test_head_dim_calculation(self):
        """Test head dimension calculation"""
        config = CacaConfig(hidden_size=1024, num_attention_heads=8)
        assert config.head_dim == 128
    
    def test_gqa_ratio(self):
        """Test GQA ratio"""
        config = CacaConfig(num_attention_heads=16, num_key_value_heads=4)
        gqa_ratio = config.num_attention_heads // config.num_key_value_heads
        assert gqa_ratio == 4
    
    def test_sliding_window_validation(self):
        """Test sliding window validation"""
        # Valid: sliding_window < max_position_embeddings
        config = CacaConfig(
            max_position_embeddings=8192,
            sliding_window=4096
        )
        assert config.sliding_window == 4096
        
        # Invalid: sliding_window > max_position_embeddings
        with pytest.raises(ValueError):
            CacaConfig(
                max_position_embeddings=2048,
                sliding_window=4096
            )
    
    def test_dropout_range(self):
        """Test dropout parameter ranges"""
        # Valid dropouts
        config = CacaConfig(attention_dropout=0.1, hidden_dropout=0.1)
        assert config.attention_dropout == 0.1
        
        # Invalid dropout (out of range)
        with pytest.raises(ValueError):
            CacaConfig(attention_dropout=1.5)
        
        with pytest.raises(ValueError):
            CacaConfig(attention_dropout=-0.1)
    
    def test_config_serialization(self):
        """Test config to/from dict"""
        config = CacaConfig(
            vocab_size=50000,
            hidden_size=1024,
        )
        
        # To dict
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict['vocab_size'] == 50000
        
        # From dict
        new_config = CacaConfig(**config_dict)
        assert new_config.vocab_size == config.vocab_size
        assert new_config.hidden_size == config.hidden_size
    
    def test_config_save_load(self, tmp_path):
        """Test config save and load"""
        config = CacaConfig(vocab_size=50000)
        
        # Save
        save_path = tmp_path / "config"
        save_path.mkdir()
        config.save_pretrained(str(save_path))
        
        # Load
        loaded_config = CacaConfig.from_pretrained(str(save_path))
        assert loaded_config.vocab_size == config.vocab_size
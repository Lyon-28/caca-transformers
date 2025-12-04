"""Tests for Caca modeling"""

import pytest
import torch
from caca_transformers import CacaConfig, CacaForCausalLM, CacaModel


class TestCacaModeling:
    """Test suite for Caca models"""
    
    @pytest.fixture
    def small_config(self):
        """Small config for fast testing"""
        return CacaConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=256,
            max_position_embeddings=512,
            sliding_window=128,
        )
    
    def test_config_creation(self, small_config):
        """Test config creation"""
        assert small_config.vocab_size == 1000
        assert small_config.hidden_size == 128
        assert small_config.head_dim == 32
    
    def test_model_creation(self, small_config):
        """Test model creation"""
        model = CacaModel(small_config)
        assert model is not None
        assert model.config.vocab_size == 1000
    
    def test_causal_lm_creation(self, small_config):
        """Test CausalLM creation"""
        model = CacaForCausalLM(small_config)
        assert model is not None
        assert hasattr(model, 'lm_head')
    
    def test_forward_pass(self, small_config):
        """Test forward pass"""
        model = CacaForCausalLM(small_config)
        model.eval()
        
        batch_size, seq_length = 2, 64
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_length))
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        assert outputs['logits'].shape == (batch_size, seq_length, small_config.vocab_size)
    
    def test_kv_cache(self, small_config):
        """Test KV cache functionality"""
        model = CacaForCausalLM(small_config)
        model.eval()
        
        batch_size, seq_length = 2, 64
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_length))
        
        with torch.no_grad():
            # First pass with cache
            outputs = model(input_ids, use_cache=True)
            past_kv = outputs['past_key_values']
            
            # Check cache structure
            assert past_kv is not None
            assert len(past_kv) == small_config.num_hidden_layers
            
            # Next token with cache
            next_token = torch.randint(0, small_config.vocab_size, (batch_size, 1))
            outputs_next = model(next_token, past_key_values=past_kv, use_cache=True)
            
            assert outputs_next['logits'].shape == (batch_size, 1, small_config.vocab_size)
    
    def test_generation(self, small_config):
        """Test generation capability"""
        model = CacaForCausalLM(small_config)
        model.eval()
        
        batch_size = 1
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, 10))
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=20,
                do_sample=False,
                use_cache=True,
            )
        
        assert outputs.shape[0] == batch_size
        assert outputs.shape[1] == 20
    
    def test_gradient_checkpointing(self, small_config):
        """Test gradient checkpointing"""
        model = CacaForCausalLM(small_config)
        model.gradient_checkpointing_enable()
        
        assert model.model.gradient_checkpointing == True
    
    @pytest.mark.parametrize("batch_size,seq_length", [
        (1, 32),
        (2, 64),
        (4, 128),
    ])
    def test_different_batch_sizes(self, small_config, batch_size, seq_length):
        """Test with different batch sizes and sequence lengths"""
        model = CacaForCausalLM(small_config)
        model.eval()
        
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_length))
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        assert outputs['logits'].shape == (batch_size, seq_length, small_config.vocab_size)
    
    def test_attention_mask(self, small_config):
        """Test with attention mask"""
        model = CacaForCausalLM(small_config)
        model.eval()
        
        batch_size, seq_length = 2, 32
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        attention_mask[:, -5:] = 0  # Mask last 5 tokens
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        
        assert outputs['logits'].shape == (batch_size, seq_length, small_config.vocab_size)
    
    def test_labels_loss(self, small_config):
        """Test loss computation with labels"""
        model = CacaForCausalLM(small_config)
        model.train()
        
        batch_size, seq_length = 2, 32
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_length))
        labels = input_ids.clone()
        
        outputs = model(input_ids, labels=labels)
        
        assert outputs['loss'] is not None
        assert outputs['loss'].item() > 0
    
    def test_model_size(self, small_config):
        """Test model parameter count"""
        model = CacaForCausalLM(small_config)
        total_params = sum(p.numel() for p in model.parameters())
        
        assert total_params > 0
        print(f"\nSmall model params: {total_params:,}")
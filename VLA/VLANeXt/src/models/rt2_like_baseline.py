import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    SiglipVisionModel, SiglipImageProcessor, LlamaForCausalLM, 
    LogitsProcessor, LogitsProcessorList
)
from PIL import Image

from .encoder import ActionTransformerProjector


class LlamaProcessorWrapper:
    def __init__(self, tokenizer, image_processor):
        self.tokenizer = tokenizer
        self.image_processor = image_processor

class RT2LikeBaseline(nn.Module):
    def __init__(
        self, 
        lmm_path="meta-llama/Llama-3.2-1B-Instruct",
        vision_encoder_path="google/siglip2-base-patch16-256",
        action_dim=7,
        num_actions=1,
        num_history=0,
        use_proprio_input_vlm=True,
        use_transformer_projector=True,
        projector_depth=2,
        projector_num_heads=4,
        backbone_mode="finetune", # Options: "frozen", "finetune"
        gradient_checkpointing=True,
        num_bins=256,
        attn_implementation="flash_attention_2",
    ):
        super().__init__()
        
        print(f"Initializing RT-2 Baseline with attn_implementation: {attn_implementation}")

        self.lmm = LlamaForCausalLM.from_pretrained(
            lmm_path, dtype=torch.bfloat16, attn_implementation=attn_implementation
        )
        self.vision_encoder = SiglipVisionModel.from_pretrained(
            vision_encoder_path, dtype=torch.bfloat16, attn_implementation=attn_implementation
        )
        
        tokenizer = AutoTokenizer.from_pretrained(lmm_path)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        image_processor = SiglipImageProcessor.from_pretrained(vision_encoder_path)
        self.processor = LlamaProcessorWrapper(tokenizer, image_processor)

        self.hidden_size = self.lmm.config.hidden_size
        self.vision_projector = nn.Sequential(
            nn.Linear(self.vision_encoder.config.hidden_size, self.hidden_size),
            nn.SiLU(), 
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(), 
            nn.Linear(self.hidden_size, self.hidden_size)
        )

        if backbone_mode == "frozen":
            self.lmm.requires_grad_(False)
            self.vision_encoder.requires_grad_(False)
        elif backbone_mode == "finetune":
            self.lmm.requires_grad_(True)
            self.vision_encoder.requires_grad_(True)
        else:
            raise ValueError(f"Unknown backbone_mode: {backbone_mode}")

        if gradient_checkpointing:
            if hasattr(self.lmm, "gradient_checkpointing_enable"):
                self.lmm.gradient_checkpointing_enable()
            if hasattr(self.lmm, "enable_input_require_grads"):
                self.lmm.enable_input_require_grads()
            if hasattr(self.lmm.config, "use_cache"):
                self.lmm.config.use_cache = False
            if hasattr(self.vision_encoder, "gradient_checkpointing_enable"):
                self.vision_encoder.gradient_checkpointing_enable()

        self.action_dim = action_dim
        self.num_actions = num_actions
        self.num_history = num_history
        self.num_bins = num_bins
        self.use_proprio_input_vlm = use_proprio_input_vlm

        if self.use_proprio_input_vlm:
            if use_transformer_projector:
                self.action_projector = ActionTransformerProjector(
                    action_dim=action_dim,
                    hidden_size=self.hidden_size,
                    depth=projector_depth,
                    num_heads=projector_num_heads
                )
            else:
                self.action_projector = nn.Linear(action_dim, self.hidden_size)
        else:
            self.action_projector = None

        vocab_limit = 128000
        self.action_token_start_idx = vocab_limit - num_bins
        print(f"RT-2 Mode: Using vocabulary indices [{self.action_token_start_idx}, {vocab_limit}) for {num_bins} action bins.")

    def forward(self, input_ids=None, attention_mask=None, actions=None, proprioception=None, proprio_attention_mask=None, pixel_values=None, **kwargs):
        vision_outputs = self.vision_encoder(pixel_values, output_hidden_states=True)
        image_feats = vision_outputs.last_hidden_state
        image_embeds = self.vision_projector(image_feats)

        B = input_ids.shape[0]

        if image_embeds.shape[0] != B:
            num_views = image_embeds.shape[0] // B
            image_embeds = image_embeds.view(B, num_views, -1, image_embeds.shape[-1])
            image_embeds = image_embeds.flatten(1, 2)

        text_embeds = self.lmm.model.embed_tokens(input_ids)

        proprio_embeds = None
        if self.use_proprio_input_vlm and proprioception is not None:
            proprio_embeds = self.action_projector(proprioception.to(device=text_embeds.device, dtype=text_embeds.dtype))

        actions_flat = actions.view(B, -1)
        
        gt_actions_clamped = torch.clamp(actions_flat, -1, 1)
        gt_actions_idx = ((gt_actions_clamped + 1) / 2 * (self.num_bins - 1)).round().long()
        
        action_input_ids = gt_actions_idx + self.action_token_start_idx
        action_embeds = self.lmm.model.embed_tokens(action_input_ids)

        embeds_list = [image_embeds]
        
        image_mask = torch.ones(B, image_embeds.shape[1], device=attention_mask.device, dtype=attention_mask.dtype)
        mask_list = [image_mask]
        
        labels_list = [torch.full((B, image_embeds.shape[1]), -100, dtype=torch.long, device=attention_mask.device)]

        if proprio_embeds is not None:
            embeds_list.append(proprio_embeds)
            
            p_len = proprio_embeds.shape[1]
            if proprio_attention_mask is not None:
                mask_list.append(proprio_attention_mask.to(attention_mask.device))
            else:
                p_mask = torch.ones(B, p_len, device=attention_mask.device, dtype=attention_mask.dtype)
                mask_list.append(p_mask)
            
            labels_list.append(torch.full((B, p_len), -100, dtype=torch.long, device=attention_mask.device))
        
        embeds_list.append(text_embeds)
        mask_list.append(attention_mask)
        labels_list.append(torch.full((B, text_embeds.shape[1]), -100, dtype=torch.long, device=attention_mask.device))

        embeds_list.append(action_embeds)
        action_mask = torch.ones(B, action_embeds.shape[1], device=attention_mask.device, dtype=attention_mask.dtype)
        mask_list.append(action_mask)
        labels_list.append(action_input_ids)

        inputs_embeds = torch.cat(embeds_list, dim=1)
        combined_attention_mask = torch.cat(mask_list, dim=1)
        combined_labels = torch.cat(labels_list, dim=1)

        outputs = self.lmm(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_attention_mask,
            labels=combined_labels,
        )
        
        return outputs.loss

    @torch.no_grad()
    def predict_action(self, input_ids, attention_mask, proprioception=None, proprio_attention_mask=None, pixel_values=None, **kwargs):
        B = input_ids.shape[0]

        vision_outputs = self.vision_encoder(pixel_values, output_hidden_states=True)
        image_embeds = self.vision_projector(vision_outputs.last_hidden_state)

        if image_embeds.shape[0] != B:
            num_views = image_embeds.shape[0] // B
            image_embeds = image_embeds.view(B, num_views, -1, image_embeds.shape[-1])
            image_embeds = image_embeds.flatten(1, 2)

        text_embeds = self.lmm.model.embed_tokens(input_ids)

        embeds_list = [image_embeds]
        mask_list = [torch.ones(B, image_embeds.shape[1], device=attention_mask.device, dtype=attention_mask.dtype)]
        
        if self.use_proprio_input_vlm and proprioception is not None:
            proprio_embeds = self.action_projector(proprioception)
            embeds_list.append(proprio_embeds)
            if proprio_attention_mask is not None:
                mask_list.append(proprio_attention_mask.to(attention_mask.device))
            else:
                mask_list.append(torch.ones(B, proprio_embeds.shape[1], device=attention_mask.device, dtype=attention_mask.dtype))
        
        embeds_list.append(text_embeds)
        mask_list.append(attention_mask)

        inputs_embeds = torch.cat(embeds_list, dim=1)
        attention_mask = torch.cat(mask_list, dim=1)

        class RT2ActionLogitsProcessor(LogitsProcessor):
            def __init__(self, start_idx, end_idx):
                self.start_idx = start_idx
                self.end_idx = end_idx
            
            def __call__(self, input_ids, scores):
                scores[:, :self.start_idx] = float('-inf')
                scores[:, self.end_idx:] = float('-inf')
                return scores

        logits_processor = LogitsProcessorList([
            RT2ActionLogitsProcessor(self.action_token_start_idx, self.action_token_start_idx + self.num_bins)
        ])
        
        total_action_tokens = self.num_actions * self.action_dim

        pad_token_id = getattr(self.lmm.config, "pad_token_id", None)
        pad_token_id = pad_token_id if pad_token_id is not None else 0

        dummy_input_ids = torch.full(
            (B, inputs_embeds.shape[1]), 
            pad_token_id,
            dtype=torch.long, 
            device=inputs_embeds.device
        )

        outputs = self.lmm.generate(
            input_ids=dummy_input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=total_action_tokens,
            logits_processor=logits_processor,
            do_sample=False,
            use_cache=True,
            eos_token_id=[],
            pad_token_id=pad_token_id
        )

        generated_ids = outputs[:, inputs_embeds.shape[1]:]

        bin_indices = generated_ids - self.action_token_start_idx
        
        action_flat = (bin_indices.float() / (self.num_bins - 1)) * 2 - 1
        action = action_flat.view(B, self.num_actions, self.action_dim).to(dtype=self.lmm.dtype)
        
        return action


if __name__ == "__main__":
    print("Testing RT-2 Like Baseline...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    
    # Initialize Model
    model = RT2LikeBaseline(
        lmm_path="meta-llama/Llama-3.2-1B-Instruct",
        vision_encoder_path="google/siglip2-base-patch16-256",
        action_dim=7, num_actions=8, num_history=8,
        backbone_mode="finetune", gradient_checkpointing=False
    ).to(device, dtype)

    B = 2
    # Dummy Data
    img = Image.new('RGB', (256, 256), color='red')
    imgs = [img] * B
    
    # Process images
    pixel_values = torch.stack([
        model.processor.image_processor(i, return_tensors="pt")["pixel_values"].squeeze(0) for i in imgs
    ]).to(device, dtype)
    
    # Process text
    texts = ["Pick up the red block."] * B
    text_inputs = model.processor.tokenizer(texts, padding=True, return_tensors="pt")
    input_ids = text_inputs["input_ids"].to(device)
    attention_mask = text_inputs["attention_mask"].to(device)
    
    # Dummy tensors
    act_gt = torch.randn(B, 8, 7, device=device, dtype=dtype)
    proprio = torch.randn(B, 8, 7, device=device, dtype=dtype)

    # Test forward (training)
    loss = model(input_ids=input_ids, attention_mask=attention_mask, actions=act_gt,
                 proprioception=proprio, pixel_values=pixel_values)
    print(f"Training Loss: {loss.item():.4f}")
    
    # Test predict_action (inference)
    pred = model.predict_action(input_ids=input_ids, attention_mask=attention_mask,
                                proprioception=proprio, pixel_values=pixel_values)
    print(f"Predicted Action Shape: {pred.shape}")
    
    print("\nTest Passed!")
import math
from typing import List, Optional
import json
import timm
import torch
import torchvision
from PIL import Image
from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from torchvision import transforms

from .configuration_minicpm import MiniCPMVConfig
from .modeling_minicpm import MiniCPMForCausalLM, MiniCPMPreTrainedModel
from .resampler import Resampler


class MiniCPMVPreTrainedModel(MiniCPMPreTrainedModel):
    config_class = MiniCPMVConfig


class MiniCPMV(MiniCPMVPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.llm = MiniCPMForCausalLM(config)
        self.vpm = self.init_vision_module()
        self.vision_dim = self.vpm.embed_dim
        self.embed_dim = self.llm.config.hidden_size
        self.resampler = self.init_resampler(self.embed_dim, self.vision_dim)
        self.transform = self.init_transform()

    def init_vision_module(self):
        model = timm.create_model(
            self.config.vision_encoder,
            pretrained=False,
            num_classes=0,
            dynamic_img_size=True,
            dynamic_img_pad=True
        )

        if isinstance(model, timm.models.VisionTransformer):
            if model.attn_pool is not None:
                model.attn_pool = torch.nn.Identity()

        if self.config.drop_vision_last_layer:
            model.blocks = model.blocks[:-1]

        return model

    def init_resampler(self, embed_dim, vision_dim):
        return Resampler(
            grid_size=int(math.sqrt(self.config.query_num)),
            embed_dim=embed_dim,
            num_heads=embed_dim // 128,
            kv_dim=vision_dim,
            adaptive=True
        )

    def init_transform(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD
                ),
            ]
        )

    def get_input_embeddings(self):
        return self.llm.embed_tokens

    def set_input_embeddings(self, value):
        self.llm.embed_tokens = value

    def get_output_embeddings(self):
        return self.llm.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.llm.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.llm = decoder

    def get_decoder(self):
        return self.llm

    def get_vision_embedding(self, pixel_values):
        res = []
        dtype = self.vpm.pos_embed.data.dtype
        for pixel_value in pixel_values:
            H, W = pixel_value.shape[-2:]
            tgt_size = (
            math.ceil(H / self.vpm.patch_embed.patch_size[0]), math.ceil(W / self.vpm.patch_embed.patch_size[0]))
            vision_embedding = self.vpm.forward_features(pixel_value.unsqueeze(0).type(dtype))
            if hasattr(self.vpm, 'num_prefix_tokens') and self.vpm.num_prefix_tokens > 0:
                vision_embedding = vision_embedding[:, self.vpm.num_prefix_tokens:]
            res.append(self.resampler(vision_embedding, tgt_size))
        return torch.vstack(res)

    def get_vllm_embedding(self, data):
        if "vision_hidden_states" not in data:
            pixel_values_list = data["pixel_values"]
            vision_hidden_states = []
            for pixel_values in pixel_values_list:
                if len(pixel_values) > 0:
                    vision_hidden_states.append(self.get_vision_embedding(pixel_values))
                elif self.training:
                    dtype = self.vpm.pos_embed.data.dtype
                    device = self.vpm.pos_embed.data.device
                    dummy_image = torch.zeros(
                        (1, 3, 224, 224), device=device, dtype=dtype
                    )
                    vision_hidden_states.append(self.get_vision_embedding(dummy_image))
                else:
                    vision_hidden_states.append([])

        else:
            vision_hidden_states = data["vision_hidden_states"]

        vllm_embedding = (
            self.llm.model.embed_tokens(data["input_ids"]) * self.llm.config.scale_emb
        )
        vision_hidden_states = [
            i.type(vllm_embedding.dtype) if isinstance(i, torch.Tensor) else i
            for i in vision_hidden_states
        ]

        bs = len(data["input_ids"])
        for i in range(bs):
            cur_vs_hs = vision_hidden_states[i]
            if len(cur_vs_hs) > 0:
                cur_vllm_emb = vllm_embedding[i]
                cur_image_bound = data["image_bounds"][i]
                if len(cur_image_bound) > 0:
                    image_indices = torch.stack(
                        [
                            torch.arange(r[0], r[1], dtype=torch.long)
                            for r in cur_image_bound
                        ]
                    ).to(vllm_embedding.device)

                    cur_vllm_emb.scatter_(
                        0,
                        image_indices.view(-1, 1).repeat(1, cur_vllm_emb.shape[-1]),
                        cur_vs_hs.view(-1, cur_vs_hs.shape[-1]),
                    )
                elif self.training:
                    cur_vllm_emb += cur_vs_hs[0].mean() * 0

        return vllm_embedding, vision_hidden_states

    def forward(self, data, **kwargs):
        vllm_embedding, vision_hidden_states = self.get_vllm_embedding(data)
        position_ids = data["position_ids"]
        if position_ids.dtype != torch.int64:
            position_ids = position_ids.long()

        return self.llm(
            input_ids=None,
            position_ids=position_ids,
            inputs_embeds=vllm_embedding,
            **kwargs
        )

    def _decode_text(self, result_ids, tokenizer):
        result_text = []
        for result in result_ids:
            result = result[result != 0]
            if result[0] == tokenizer.bos_id:
                result = result[1:]
            if result[-1] == tokenizer.eos_id:
                result = result[:-1]
            result_text.append(tokenizer.decode(result).strip())
        return result_text

    def _decode(self, inputs_embeds, tokenizer, **kwargs):
        output = self.llm.generate(
            inputs_embeds=inputs_embeds,
            pad_token_id=0,
            eos_token_id=tokenizer.eos_token_id if tokenizer is not None else kwargs.pop("eos_token_id", 2),
            **kwargs
        )
        return output

    def generate(
        self,
        input_ids,
        pixel_values=None,
        image_sizes=[],
        image_bounds=[],
        tgt_sizes=[],
        tokenizer=None,
        vision_hidden_states=None,
        **kwargs
    ):
        bs = len(input_ids)
        img_list = pixel_values
        if img_list == None:
            img_list = [[] for i in range(bs)]
        assert bs == len(img_list)

        if vision_hidden_states is None:
            pixel_values = []
            for i in range(bs):
                img_inps = []
                for img in img_list[i]:
                    img_inps.append(img.to(self.device, self.dtype))
                pixel_values.append(img_inps)

        # with torch.inference_mode():
        (
            input_embeds,
            vision_hidden_states,
        ) = self.get_vllm_embedding({
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "image_sizes": image_sizes,
            "image_bounds": image_bounds,
            "tgt_sizes": tgt_sizes
        })
        result = self._decode(input_embeds, tokenizer, **kwargs)

        return result

    def chat(
        self,
        image,
        msgs,
        context,
        tokenizer,
        processor,
        vision_hidden_states=None,
        max_new_tokens=1024,
        sampling=True,
        max_inp_length=2048,
        **kwargs
    ):
        if isinstance(msgs, str):
            msgs = json.loads(msgs)
        # msgs to prompt

        prompt = processor.tokenizer.apply_chat_template(msgs)
        inputs = processor(prompt, [image], return_tensors="pt").to(self.device)

        if sampling:
            generation_config = {
                "top_p": 0.8,
                "top_k": 100,
                "temperature": 0.7,
                "do_sample": True,
                "repetition_penalty": 1.05
            }
        else:
            generation_config = {
                "num_beams": 3,
                "repetition_penalty": 1.2,
            }

        generation_config.update(
            (k, kwargs[k]) for k in generation_config.keys() & kwargs.keys()
        )
        with torch.inference_mode():
            res = self.generate(
                **inputs,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                vision_hidden_states=vision_hidden_states,
                **generation_config,
            )
            res = self._decode_text(res, tokenizer)
        answer = res[0]
        context = msgs.copy()
        context.append({"role": "assistant", "content": answer})

        return answer, context, generation_config
    

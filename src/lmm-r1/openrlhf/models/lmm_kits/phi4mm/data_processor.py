from typing import List, Dict, Union
from ..base.data_processor import BaseDataProcessor
import torch
import os

class Phi4MMDataProcessor(BaseDataProcessor):
    def __call__(
        self,
        messages,
        max_length,
        padding=True,
        device=None,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
    ):
        texts = self.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs = self.get_images_from_messages(messages)
        # currently only support image and text
        batch = self.processor(
            text=texts,
            images=image_inputs,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        if device:
            return {k: v.to(device) for k, v in batch.items() if v is not None and v.numel() > 0}
        return batch
    
    def apply_chat_template(
        self,
        messages: Union[Dict, List[str], str],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> List[str]:
        messages = self._format_messages(messages)
        messages = self._convert_message_format(messages)
        return self.processor.tokenizer.apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt
        )

    def _convert_message_format(self, messages: List[List[Dict]]) -> List[List[Dict]]:
        converted_messages = []
        
        for message in messages:
            new_message = []
            image_counter = 1
            for msg in message:
                role = msg["role"]
                content = msg["content"]

                # Process the content to combine text and images
                processed_content = ""
                
                if isinstance(content,list):
                    for content_item in content:
                        if content_item["type"] == "text":
                            processed_content += content_item["text"]
                        elif content_item["type"] == "image":
                            image_placeholder = f"<|image_{image_counter}|>"
                            processed_content += image_placeholder
                            image_counter += 1
                else:
                    processed_content += content

                new_message.append({"role": role, "content": processed_content})
            converted_messages.append(new_message)
        return converted_messages

    def make_input_batch(self, inputs: List[Dict]) -> Dict:
        # each element has no batch dimension
        batch = {}
        # collect all keys
        for inp in inputs:
            batch.update({k:None for k,v in inp.items() if v is not None})
        
        if "input_image_embeds" in batch.keys():
            max_crops = max([inp["input_image_embeds"].size(1) for inp in inputs if "input_image_embeds" in inp])

        for k in batch.keys():
            if k in ["input_ids", "attention_mask"]:
                batch[k] = torch.stack([inp[k] for inp in inputs if k in inp], dim=0)
            elif k == "input_mode":
                batch[k] = torch.stack([inp[k] for inp in inputs if k in inp], dim=0)
                # text mode = 0, vision mode = 1
                batch[k] = batch[k].max().unsqueeze(0)
                if batch[k][0] > 1:
                    raise ValueError("Only vision mode and text mode are supported")
            elif k == "input_image_embeds":
                #first dimension is image number, second dimension is crop number
                batch_list = []
                for inp in inputs:
                    batch_list.extend(list(inp[k]))
                batch_list = [self.processor.image_processor.pad_to_max_num_crops(im, max_crops) for im in batch_list]
                batch[k] = torch.stack(batch_list, dim=0)
            elif k == "image_attention_mask":
                batch_list = []
                for inp in inputs:
                    batch_list.extend(list(inp[k]))
                batch_list = [self.processor.image_processor.pad_mask_to_max_num_crops(mask, max_crops) for mask in batch_list]
                batch[k] = torch.stack(batch_list, dim=0)
            else:
                # concat all items in a batch in the first dimension
                batch[k] = torch.cat([inp[k] for inp in inputs if k in inp], dim=0)

        return batch
    
    def split_input_batch(self, batch: Dict) -> List[Dict]:
        batch_size = len(batch["input_ids"])
        batch_kwargs = [{} for _ in range(batch_size)]
        # first process None values
        keys = []
        for k, v in batch.items():
            if v is not None:
                keys.append(k)
            else:
                for i in range(batch_size):
                    batch_kwargs[i][k] = None

        if "input_image_embeds" in keys and ("input_ids" not in keys or "num_img_tokens" not in keys):
            raise ValueError("Cannot split batch with pixel_values without input_ids and num_img_tokens")
        
        for k in ["input_ids", "attention_mask"]:
            if k in keys:
                vals = batch[k]
                if isinstance(vals, torch.Tensor):
                    vals = torch.unbind(vals)
                assert batch_size == len(vals)
                for i, v in enumerate(vals):
                    batch_kwargs[i][k] = v
        #fill input mode
        for i in range(batch_size):
            batch_kwargs[i]["input_mode"] = torch.tensor([0], dtype=torch.long)

        if "input_image_embeds" in keys:
            input_image_embeds = batch["input_image_embeds"]
            image_sizes = batch["image_sizes"]
            image_attention_mask = batch["image_attention_mask"]
            num_img_tokens = batch["num_img_tokens"]

            image_sizes = list(image_sizes)
            input_image_embeds = list(input_image_embeds)
            image_attention_mask = list(image_attention_mask)
            num_img_tokens = list(num_img_tokens)

            num_img_tokens_per_image = num_img_tokens
            # Split pixel values and image sizes for each sample. Each sample can have multiple images.
            image_token_id = 200010 #vllm use 200010 as image token
            for i in range(batch_size):
                input_ids_i = batch_kwargs[i]["input_ids"]
                if not isinstance(input_ids_i, torch.Tensor):
                    input_ids_i = torch.tensor(input_ids_i)
                rest_image_token_num = (input_ids_i == image_token_id).sum().item()
                if rest_image_token_num == 0:
                    batch_kwargs[i]["input_image_embeds"] = None
                    batch_kwargs[i]["image_sizes"] = None
                    batch_kwargs[i]["image_attention_mask"] = None
                    batch_kwargs[i]["num_img_tokens"] = None
                    batch_kwargs[i]["input_mode"] = torch.tensor([0], dtype=torch.long)
                    continue
                image_sizes_i = []
                input_image_embeds_i = []
                image_attention_mask_i = []
                num_img_tokens_i = []
                while rest_image_token_num > 0:
                    if len(num_img_tokens_per_image) == 0:
                        raise ValueError("Mismatch in total number of image tokens")
                    cur_num_img_tokens = num_img_tokens_per_image.pop(0)
                    image_sizes_i.append(image_sizes.pop(0))
                    input_image_embeds_i.append(input_image_embeds.pop(0))
                    image_attention_mask_i.append(image_attention_mask.pop(0))
                    num_img_tokens_i.append(cur_num_img_tokens)
                    rest_image_token_num -= cur_num_img_tokens.item()
                assert rest_image_token_num == 0, "Mismatch in total number of image tokens"
                batch_kwargs[i]["input_image_embeds"] = torch.stack(input_image_embeds_i, dim=0)
                batch_kwargs[i]["image_sizes"] = torch.stack(image_sizes_i, dim=0)
                batch_kwargs[i]["image_attention_mask"] = torch.stack(image_attention_mask_i, dim=0)
                batch_kwargs[i]["num_img_tokens"] = torch.stack(num_img_tokens_i, dim=0)
                batch_kwargs[i]["input_mode"] = torch.tensor([1], dtype=torch.long)
            assert len(image_sizes) == 0
            assert len(input_image_embeds) == 0
            assert len(image_attention_mask) == 0
            assert len(num_img_tokens) == 0

        return batch_kwargs

DataProcessor = Phi4MMDataProcessor

__all__ = ["DataProcessor"]
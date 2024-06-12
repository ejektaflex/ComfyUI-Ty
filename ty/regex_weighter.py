import folder_paths
import comfy.utils
import comfy.lora
import re


class LoraBlockRegexLoader:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",),
                             "clip": ("CLIP",),
                             "lora": (folder_paths.get_filename_list("loras"), ),
                             "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                             "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                             "regex_map": ("STRING", {"multiline": True, "placeholder": "weight map", "default": ".*|1", "pysssss.autocomplete": False}),
                            }
                }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "doLoad"

    CATEGORY = "Ty/BlockWeight"

    @staticmethod
    def load_lora_for_models(model, clip, lora, strength_model, strength_clip, regex_map):
        key_map = comfy.lora.model_lora_keys_unet(model.model)
        key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)
        loaded = comfy.lora.load_lora(lora, key_map)

        reg_list = [s.split('|') for s in regex_map.splitlines() if s.strip()]

        cloned_model = model.clone()
        weight_mapping = {}

        for reg in reg_list:
            if len(reg) != 2:
                raise ValueError("Line '" + reg + "' must have two values separated by a pipe!")

        print("Block Weights:")
        for k, v in loaded.items():
            weight_value = 0.0

            for reg in reg_list:
                try:
                    re.compile(reg[0])
                except re.error:
                    raise ValueError("Regex " + reg[0] + " is not a valid regex!")
                if re.search(reg[0], k):
                    print("Overriding weight: " + k)
                    weight_value = float(reg[1])
                    break

            cloned_model.add_patches({k: v}, strength_model * weight_value)

        cloned_clip = clip.clone()
        cloned_clip.add_patches(loaded, strength_clip)

        print("Weight mapping:")
        print(weight_mapping)

        return (cloned_model, cloned_clip)



    def doLoad(self, model, clip, lora, strength_model, strength_clip, regex_map):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)
        
        if len(regex_map.strip()) == 0:
            return (model, clip)

        lora_path = folder_paths.get_full_path("loras", lora)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = LoraBlockRegexLoader.load_lora_for_models(model, clip, lora, strength_model, strength_clip, regex_map)
        return (model_lora, clip_lora)





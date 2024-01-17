import PIL
import traceback
from datetime import datetime
import os
import logging
import torch
import numpy as np
import json

class DebugDumper:

    _dumpers = {}

    @staticmethod
    def get_config(debug_dumper_name:str, parameter_name:str):

        config_file_name = 'config.json'

        current_script_path = os.path.abspath(__file__)
        current_script_dir = os.path.dirname(current_script_path)
        parent_dir = os.path.dirname(current_script_dir)
        config_file_name = os.path.join(parent_dir, config_file_name)

        if os.path.exists(config_file_name):
            with open(config_file_name) as json_file:
                config = json.load(json_file)
                if debug_dumper_name in config:
                    if parameter_name in config[debug_dumper_name]:
                        return config[debug_dumper_name][parameter_name]
                    else:
                        return None
        else:
            print(f"DebugDumper: can't find config file {config_file_name}")

        return None

    @staticmethod
    def GetByName(name: str, base_path: str = None, create_subdir: bool = True, vae = None) -> 'DebugDumper':

        if name not in DebugDumper._dumpers:
            DebugDumper._dumpers[name] = DebugDumper._create(name, base_path, create_subdir=create_subdir, vae=vae)

        level = DebugDumper.get_config(name, 'level')
        if level is None:
            level = 0

        DebugDumper._dumpers[name].level = level

        return DebugDumper._dumpers[name]

    @staticmethod
    def _create(name: str, base_path: str, create_subdir: bool = True, vae = None):

        self = DebugDumper()

        self.vae = vae

        if base_path is not None:
            self.base_path = base_path
        else:
            self.base_path = os.getcwd()

        if create_subdir:
            self.base_path = os.path.join(self.base_path, datetime.now().strftime("%y-%m-%d_%H-%M-%S"))
        os.makedirs(self.base_path)

        print(f'Dir {self.base_path} created')

        if name is not None:
            self.name = name
        else:
            self.name = 'DebugDumper'

        local_logger = logging.getLogger(self.name)
        local_logger.setLevel(logging.INFO)

        # Create a formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Create a file handler and set the formatter
        file_handler = logging.FileHandler('myapp.log')
        file_handler.setFormatter(formatter)

        # Create a console handler and set the formatter
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # Add the handlers to the logger
        local_logger.addHandler(file_handler)
        local_logger.addHandler(console_handler)

        local_logger.info(f"DebugDumper '{self.name}' initialized")

        return self

    @staticmethod
    def _save_pytorch_tensor_as_images(tensor,
                                       filename: str,
                                       is_normalized='auto',
                                       local_logger = None,
                                       metadata = None) -> bool:

        shape = tensor.shape

        if len(shape) == 2:

            tensor = tensor.cpu()

            if is_normalized == True or (is_normalized == 'auto' and torch.min(tensor) < 0):
                tensor = tensor / 2 + 0.5     #denormalization
                filename += '_denorm'

            numpy_array = tensor.cpu().numpy()
            numpy_array = np.uint8(255 * numpy_array)

            pil_image = PIL.Image.fromarray(numpy_array, mode='L')

            if metadata is not None:
                pil_image.info["Description"] = json.dumps(metadata)

            filename += '.png'
            pil_image.save(filename)

            if local_logger is not None:
                local_logger.info(f"\tSaved to {filename}")

            return True

        elif len(shape) == 3:

            if shape[0] > 4:
                return False
            else:
                tensor = tensor.cpu()

                if is_normalized == True or (is_normalized == 'auto' and torch.min(tensor) < 0):
                    tensor = tensor / 2 + 0.5  # denormalization
                    filename += '_denorm'

                numpy_array = tensor.cpu().numpy()
                numpy_array = np.uint8(255 * numpy_array)

                if tensor.shape[0] == 1:
                    pil_image = PIL.Image.fromarray(numpy_array[0], mode='L')

                elif tensor.shape[0] == 2:
                    pil_image = PIL.Image.fromarray(numpy_array.transpose(1, 2, 0), mode='LA')

                elif tensor.shape[0] == 3:
                    pil_image = PIL.Image.fromarray(numpy_array.transpose(1, 2, 0), mode='RGB').convert("RGB")

                elif tensor.shape[0] == 4:
                    pil_image = PIL.Image.fromarray(numpy_array.transpose(1, 2, 0), mode='RGBA')

            if metadata is not None:
                pil_image.info["Description"] = json.dumps(metadata)

            filename += '.png'
            pil_image.save(filename)

            if local_logger is not None:
                local_logger.info(f"\tSaved to {filename}")

            return True

        elif len(shape) == 4:
            iterations = min(shape[0], 10)
            for i in range(iterations):
                DebugDumper._save_pytorch_tensor_as_images(tensor[i],
                                                           filename + '_' + str(i),
                                                           is_normalized,
                                                           metadata=metadata)

            return True

        else:
            return False

    def dump_image(self,
                   object_name: str,
                   object_value,
                   is_normalized = 'auto',
                   stack_offset: int = 0,
                   image_only = True,
                   level = 0,
                   metadata = None):

        if not hasattr(self, 'level'):
            self.level = 0

        if self.level < level:
            return

        stack = traceback.extract_stack()
        caller_frame = stack[-2+stack_offset]

        # Extract the filename and line number
        source_filename = os.path.basename(caller_frame[0]).replace('.', '_')
        line_number = caller_frame[1]

        prefix = datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f")
        line_name = str(line_number).zfill(4)

        object_file_name = prefix + '_' + source_filename + '_' + line_name + '_' + object_name

        local_logger = logging.getLogger(self.name)
        local_logger.info(f"Logging object '{object_name}' of type {type(object_value)}, line {line_number} of the file {source_filename}")

        if self.level >= level:

            full_path = os.path.join(self.base_path, object_file_name)

            if isinstance(object_value, PIL.Image.Image):

                if metadata is not None:
                    object_value.info["Description"] = json.dumps(metadata)

                object_value.save(full_path + '.png')

            elif isinstance(object_value, np.ndarray):

                local_logger.info(f"\tShape is {object_value.shape}")

                if not image_only:
                    json_path = full_path + '.json'
                    with open(json_path, 'w') as json_file:
                        dct = {}
                        dct['dtype'] = str(object_value.dtype)
                        dct['shape'] = object_value.shape
                        dct['values'] = object_value.tolist()
                        dct['metadata'] = metadata
                        json.dump(dct, json_file)
                        local_logger.info(f"\tSaved to {json_path}")

                torch_tensor = torch.from_numpy(object_value).to('cpu')
                self._save_pytorch_tensor_as_images(torch_tensor, full_path, is_normalized, local_logger, metadata=metadata)

            elif isinstance(object_value, torch.FloatTensor) or isinstance(object_value, torch.Tensor):
                object_value = object_value.to('cpu')

                local_logger.info(f"\tShape is {object_value.shape}")

                if not image_only:
                    json_path = full_path + '.json'
                    with open(json_path, 'w') as json_file:
                        dct = {}
                        dct['dtype'] = str(object_value.dtype)
                        dct['shape'] = object_value.shape
                        dct['values'] = object_value.numpy().tolist()
                        dct['metadata'] = metadata
                        json.dump(dct, json_file)
                        local_logger.info(f"\tSaved to {json_path}")

                self._save_pytorch_tensor_as_images(object_value, full_path, is_normalized, local_logger, metadata=metadata)
            else:
                local_logger.info(
                    f"\tNOT SAVED. Datatype {type(object_value)} is not supported")
                return

        else:
            local_logger.info(
                f"\tNOT SAVED because the level of dumper {self.level} < the level of operation {level}")

    def dump_latent(self, name: str, latents, condition_kwargs={}, image_only = True, level = 0, metadata=None):

        if not hasattr(self, 'level'):
            self.level = 0

        if self.level < level:
            return

        self.dump_image(name, latents, stack_offset=-1, image_only = image_only, level = level, metadata=metadata)

        local_logger = logging.getLogger(self.name)

        if latents.shape[1] != 4:
            local_logger.info(f"\tCan't decode latent. It contains {latents.shape[1]} channels but not 4")
            return

        if hasattr(self, 'vae') and self.vae is not None:
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, **condition_kwargs)
            for i, img in enumerate(image):
                self.dump_image(name + '_' + str(i) + '_dcd',
                                img,
                                stack_offset=-1,
                                is_normalized=True,
                                image_only = image_only,
                                level = level,
                                metadata=metadata)
        else:
            local_logger.info(f"\tCan't decode latent. VAE is not defined. ")


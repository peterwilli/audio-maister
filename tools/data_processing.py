import torch
import random
import torch.nn.functional as F

class AudioPreprocessing:
    def random_scale(self, batch, keys, min=0.1, max=1):
        random_scale = torch.zeros_like(batch[keys[0]])
        random_values = torch.empty(random_scale.shape[0], device=random_scale.device, dtype=random_scale.dtype).uniform_(min, max)
        random_scale += random_values.view(-1, * [1] * (random_scale.dim() - 1))
        for key in keys:
            batch[key] = batch[key] * random_scale

    def random_shuffle(self, batch, keys):
        batch_size = batch[keys[0]].shape[0]
        random_indices = torch.randperm(batch_size)
        for key in keys:
            batch[key] = batch[key][random_indices]
                
    def normalize_tensor_per_row(self, tensor):
        shape = tensor.shape
        tensor = tensor.view(tensor.shape[0], -1)
        max_amp = tensor.max(1, keepdim=True)[0] - tensor.min(1, keepdim=True)[0]
        max_amp[max_amp == 0] = 1
        mix_scale = 1.0 / max_amp
        tensor *= mix_scale
        return tensor.view(*shape)
    
    def add_clicks(self, tensor):
        for batch_idx in range(tensor.shape[0]):
            click_range = random.randint(3, 10)
            amount = random.randint(0, 20)
            for _ in range(amount):
                random_location = random.randint(0, tensor.shape[2])
                slice = tensor[batch_idx, ..., random_location:random_location + click_range]
                random_scale = random.uniform(0, 1)
                slice[...] = torch.zeros_like(slice).uniform_(random_scale * -1, random_scale)
        return tensor

    def preprocess_train(self, batch, return_individual=False):
        batch = {
            key: self.normalize_tensor_per_row(batch[key]) for key in batch
        }
        type_target = "vocals"
        batch_size = batch[type_target].shape[0]
        # self.random_scale(batch, [type_target, type_target + '_aug_LR'], 0.5, 1)
        self.random_scale(batch, ['noise_LR'], 0.1, 1)
        self.random_scale(batch, ['effect_aug_LR', 'effect'], 0.1, 1)
        self.random_shuffle(batch, ['noise_LR'])
        self.random_shuffle(batch, ['effect_aug_LR', 'effect'])
        vocal = batch[type_target] # final target
        augLR = batch[type_target + '_aug_LR'] # augment low resolution audio
        noise = batch['noise_LR'] # augmented low resolution audio with noise
        effect_lr = batch['effect_aug_LR'] # random effect (lr)
        effect_hq = batch['effect'] # random effect (hq)
        lq_result = augLR + noise + effect_lr
        
        # if random.randint(0, 2) == 0:
        #     alpha = random.uniform(0, 1)
        #     random_noise = torch.zeros_like(vocal).uniform_(alpha * -1, alpha)
        #     lq_result += random_noise

        result = { 
            'target': self.normalize_tensor_per_row(vocal + (effect_hq * 0.5)),
            'low_quality': self.add_clicks(self.normalize_tensor_per_row(lq_result))
        }
        if return_individual:
            return result, dict(vocal=vocal, augLR=augLR, noise=noise, effect_lr=effect_lr, effect_hq=effect_hq)
        else:
            return result
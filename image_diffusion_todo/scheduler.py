from typing import Optional
import numpy as np
import torch
import torch.nn as nn

def extract(input, t: torch.Tensor, x: torch.Tensor):
    if t.ndim == 0:
        t = t.unsqueeze(0)
    shape = x.shape
    t = t.long().to(input.device)
    out = torch.gather(input, 0, t)
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)


class BaseScheduler(nn.Module):
    def __init__(self, num_train_timesteps: int, beta_1: float, beta_T: float, mode="linear"):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_timesteps = num_train_timesteps
        self.timesteps = torch.from_numpy(
            np.arange(0, self.num_train_timesteps)[::-1].copy().astype(np.int64)
        )

        if mode == "linear":
            betas = torch.linspace(beta_1, beta_T, steps=num_train_timesteps)
        elif mode == "quad":
            betas = torch.linspace(beta_1**0.5, beta_T**0.5, num_train_timesteps) ** 2
        elif mode == "cosine":
            # Nichol & Dhariwal cosine schedule
            
            s = 0.008
            t = torch.arange(num_train_timesteps + 1, dtype=torch.float32)
            f_t = torch.cos(((t / num_train_timesteps + s) / (1 + s)) * torch.pi / 2) ** 2
            alpha_bar = f_t / f_t[0]
        
            betas = []
            for i in range(1, num_train_timesteps + 1):
                betas.append(1 - alpha_bar[i] / alpha_bar[i - 1])
            betas = torch.clip(torch.tensor(betas, dtype=torch.float32), 1e-8, 0.999)
        
        else:
            raise NotImplementedError(f"{mode} is not implemented.")

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def uniform_sample_t(self, batch_size, device: Optional[torch.device] = None) -> torch.IntTensor:
        ts = np.random.choice(np.arange(self.num_train_timesteps), batch_size)
        ts = torch.from_numpy(ts)
        if device is not None:
            ts = ts.to(device)
        return ts


class DDPMScheduler(BaseScheduler):
    def __init__(self, num_train_timesteps: int, beta_1: float, beta_T: float,
                 mode="linear", sigma_type="small"):
        super().__init__(num_train_timesteps, beta_1, beta_T, mode)
        self.schedule_mode = mode      

        if sigma_type == "small":
            alphas_cumprod_t_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
            sigmas = (((1 - alphas_cumprod_t_prev) / (1 - self.alphas_cumprod)) * self.betas) ** 0.5
        elif sigma_type == "large":
            sigmas = self.betas ** 0.5
        else:
            raise ValueError("sigma_type must be 'small' or 'large'")

        self.register_buffer("sigmas", sigmas)

    # ----------------- Sampling -----------------
    def step(self, x_t: torch.Tensor, t: int, net_out: torch.Tensor, predictor: str):
        if predictor == "noise":
            return self.step_predict_noise(x_t, t, net_out)
        elif predictor == "x0":
            return self.step_predict_x0(x_t, t, net_out)
        elif predictor == "mean":
            return self.step_predict_mean(x_t, t, net_out)
        else:
            raise ValueError(f"Unknown predictor: {predictor}")

    def step_predict_noise(self, x_t: torch.Tensor, t: int, eps_theta: torch.Tensor):
        beta_t = self._get_teeth(self.betas, torch.tensor([t], device=x_t.device))
        alpha_t = self._get_teeth(self.alphas, torch.tensor([t], device=x_t.device))
        alpha_bar_t = self._get_teeth(self.alphas_cumprod, torch.tensor([t], device=x_t.device))
        alpha_bar_t_prev = self._get_teeth(
            torch.cat([torch.tensor([1.0], device=x_t.device), self.alphas_cumprod[:-1]]),
            torch.tensor([t], device=x_t.device),
        )

        mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps_theta)
        posterior_var = (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * beta_t

        if t > 0:
            noise = torch.randn_like(x_t)
            sample_prev = mean + torch.sqrt(posterior_var) * noise
        else:
            sample_prev = mean
        return sample_prev

    def step_predict_x0(self, x_t: torch.Tensor, t: int, x0_pred: torch.Tensor):
        beta_t = self._get_teeth(self.betas, torch.tensor([t], device=x_t.device))
        alpha_t = self._get_teeth(self.alphas, torch.tensor([t], device=x_t.device))
        alpha_bar_t = self._get_teeth(self.alphas_cumprod, torch.tensor([t], device=x_t.device))
        alpha_bar_t_prev = self._get_teeth(
            torch.cat([torch.tensor([1.0], device=x_t.device), self.alphas_cumprod[:-1]]),
            torch.tensor([t], device=x_t.device),
        )

        mean = (torch.sqrt(alpha_bar_t_prev) * beta_t / (1 - alpha_bar_t)) * x0_pred + \
               (torch.sqrt(alpha_t) * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) * x_t
        posterior_var = (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * beta_t

        if t > 0:
            noise = torch.randn_like(x_t)
            sample_prev = mean + torch.sqrt(posterior_var) * noise
        else:
            sample_prev = mean
        return sample_prev

    def step_predict_mean(self, x_t: torch.Tensor, t: int, mean_theta: torch.Tensor):
        beta_t = self._get_teeth(self.betas, torch.tensor([t], device=x_t.device))
        alpha_bar_t = self._get_teeth(self.alphas_cumprod, torch.tensor([t], device=x_t.device))
        alpha_bar_t_prev = self._get_teeth(
            torch.cat([torch.tensor([1.0], device=x_t.device), self.alphas_cumprod[:-1]]),
            torch.tensor([t], device=x_t.device),
        )
        posterior_var = (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * beta_t

        if t > 0:
            noise = torch.randn_like(x_t)
            sample_prev = mean_theta + torch.sqrt(posterior_var) * noise
        else:
            sample_prev = mean_theta
        return sample_prev

    # ----------------- Posterior (for training loss) -----------------
    def q_posterior(self, x0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        beta_t = self._get_teeth(self.betas, t)
        alpha_t = self._get_teeth(self.alphas, t)
        alpha_bar_t = self._get_teeth(self.alphas_cumprod, t)
        alpha_bar_t_prev = self._get_teeth(
            torch.cat([torch.tensor([1.0], device=x_t.device), self.alphas_cumprod[:-1]]), t
        )

        posterior_var = (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * beta_t
        coef_x0 = (torch.sqrt(alpha_bar_t_prev) * beta_t) / (1 - alpha_bar_t)
        coef_xt = (torch.sqrt(alpha_t) * (1 - alpha_bar_t_prev)) / (1 - alpha_bar_t)
        mean = coef_x0 * x0 + coef_xt * x_t
        return mean, posterior_var

    # ----------------- Utils -----------------
    def _get_teeth(self, consts: torch.Tensor, t: torch.Tensor):
        const = consts.gather(-1, t)
        return const.reshape(-1, 1, 1, 1)

    def add_noise(self, x_0: torch.Tensor, t: torch.IntTensor, eps: Optional[torch.Tensor] = None):
        if eps is None:
            eps = torch.randn(x_0.shape, device=x_0.device)

        alpha_bar_t = self._get_teeth(self.alphas_cumprod, t.to(self.alphas_cumprod.device))
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
        x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * eps
        return x_t, eps

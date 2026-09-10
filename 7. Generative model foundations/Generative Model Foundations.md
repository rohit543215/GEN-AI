# Generative Model Foundations — Complete Guide

---

# 1. AUTOENCODERS

## 1.1 What They Are and Why They Matter

**Autoencoders** are neural networks that learn to compress data into a lower-dimensional representation (encoding) and then reconstruct it back to its original form (decoding). They are the foundation upon which more advanced generative models (VAEs, GANs, Diffusion) are built.

**The core idea:** The network learns to copy its input through a "bottleneck" — a low-dimensional hidden layer that forces the network to capture the most important features of the data.

**Why this matters:** Once trained, the encoder can be used for dimensionality reduction, the decoder for generation, and the latent space for interpolation and manipulation.

## 1.2 The Architecture

```
Input (x) → Encoder → Latent Vector (z) → Decoder → Output (x̂)
```

- **Encoder:** Compresses input into a lower-dimensional latent representation
- **Decoder:** Reconstructs the original input from the latent representation
- **Bottleneck:** The narrowest layer (latent space) that forces information compression

## 1.3 The Math

**Reconstruction Loss** (the objective):

$$\mathcal{L} = \frac{1}{n} \sum_{i=1}^{n} ||x_i - \hat{x}_i||^2 \quad \text{(MSE for continuous data)}$$

or

$$\mathcal{L} = \frac{1}{n} \sum_{i=1}^{n} -[x_i \log(\hat{x}_i) + (1-x_i) \log(1-\hat{x}_i)] \quad \text{(BCE for binary data)}$$

**The transformation:**

$$z = f_\theta(x) \quad \text{(Encoder)}$$
$$\hat{x} = g_\phi(z) \quad \text{(Decoder)}$$

The network learns parameters $\theta$ and $\phi$ to minimize reconstruction error.

## 1.4 The Limitation (Why VAEs Exist)

**The fundamental problem:** Autoencoders learn a deterministic mapping from input to latent space. There's no regularization on the latent space — it can be discontinuous, have "holes," and be non-smooth.

**The consequence:** You can't reliably sample from the latent space to generate new data. Points between training examples often map to nonsensical outputs because the model didn't learn a continuous latent distribution.

## 1.5 Code: Autoencoder

```python
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, encoding_dim=32):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()  # For pixel values 0-1
        )
    
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat

# Training
def train_autoencoder(model, dataloader, epochs=50):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        for x, _ in dataloader:
            z, x_hat = model(x)
            loss = criterion(x_hat, x)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

## 1.6 When to Use Autoencoders

| Use Case | Why It Works |
|----------|--------------|
| **Dimensionality Reduction** | Compresses data to lower dimensions |
| **Anomaly Detection** | Anomalies have high reconstruction error |
| **Denoising** | Train to reconstruct clean data from noisy inputs |
| **Feature Extraction** | Latent representations as features |


---

# 2. VARIATIONAL AUTOENCODERS (VAE)

## 2.1 What They Are and Why They Exist

**Variational Autoencoders** fix the fundamental problem of standard autoencoders by imposing a **probability distribution** on the latent space — typically a standard Gaussian $\mathcal{N}(0, I)$.

**The core idea:** Instead of mapping input to a single point in latent space, the encoder maps it to a **distribution** (mean and variance). Sampling from this distribution during training forces the latent space to be smooth and continuous.

**Why this matters:** VAEs are **generative** — you can sample from the prior $\mathcal{N}(0, I)$ and decode to generate new, realistic data that follows the training distribution.

## 2.2 The Architecture

```
Input (x) → Encoder → μ(x), σ(x) → Sample z ~ 𝒩(μ, σ²) → Decoder → Output (x̂)
```

**Key difference from autoencoder:** The encoder outputs parameters of a distribution, not a deterministic latent vector.

## 2.3 The Math

**Encoder:** Produces parameters of the approximate posterior:

$$q_\phi(z|x) = \mathcal{N}(z; \mu_\phi(x), \sigma_\phi^2(x) I)$$

**Prior:** We want the learned distribution to match the prior:

$$p(z) = \mathcal{N}(z; 0, I)$$

**Decoder:** Reconstructs data from sampled latent:

$$p_\theta(x|z) \quad \text{(typically Gaussian or Bernoulli)}$$

**The ELBO (Evidence Lower Bound):**

$$\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) || p(z))$$

**Breaking it down:**
- **Reconstruction term:** $\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]$ — how well the decoder reconstructs the input
- **KL Divergence:** $D_{KL}(q_\phi(z|x) || p(z))$ — how close the learned distribution is to the prior

**The Reparameterization Trick:** To backpropagate through the sampling operation:

$$z = \mu + \sigma \odot \epsilon \quad \text{where } \epsilon \sim \mathcal{N}(0, I)$$

This makes the sampling differentiable.

## 2.4 The KL Divergence (Closed Form)

For Gaussian $q$ and Gaussian prior $\mathcal{N}(0, I)$:

$$D_{KL}(\mathcal{N}(\mu, \sigma^2) || \mathcal{N}(0, 1)) = -\frac{1}{2} \sum_{j=1}^{d} (1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2)$$

## 2.5 Code: VAE

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU()
        )
        
        # Latent parameters
        self.mu = nn.Linear(64, latent_dim)
        self.logvar = nn.Linear(64, latent_dim)  # Log variance for stability
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

# VAE Loss
def vae_loss(x, x_hat, mu, logvar):
    # Reconstruction loss (BCE for binary images)
    recon_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_loss
```

## 2.6 VAE vs Autoencoder

| Aspect | Autoencoder | VAE |
|--------|-------------|-----|
| **Latent space** | Deterministic, irregular | Probabilistic, smooth |
| **Generative** | No (can't sample) | Yes (sample from prior) |
| **Training** | Reconstruction only | Reconstruction + KL regularization |
| **Latent interpolation** | Not smooth | Smooth, meaningful |
| **Output** | Deterministic | Probabilistic |


---

# 3. GANS (GENERATIVE ADVERSARIAL NETWORKS)

## 3.1 What They Are

**GANs** pit two neural networks against each other in a game-theoretic framework:
- **Generator:** Creates fake data to fool the discriminator
- **Discriminator:** Tries to distinguish real from fake

**The core idea:** The competition forces both networks to improve — the generator produces increasingly realistic data, and the discriminator becomes better at detecting fakes.

## 3.2 The Architecture

```
Random Noise (z) → Generator → Fake Data
                           ↓
                    Discriminator → Real/Fake
                           ↑
Real Data → Discriminator → Real/Fake
```

## 3.3 The Math

**The Minimax Game:**

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$$

**Breaking it down:**
- **Discriminator ($D$):** Wants to maximize $\log D(x)$ (real data classified as real) and $\log(1 - D(G(z)))$ (fake data classified as fake)
- **Generator ($G$):** Wants to minimize $\log(1 - D(G(z)))$ (make fake data classified as real)

**In practice:** Generator maximizes $\log D(G(z))$ instead of minimizing $\log(1 - D(G(z)))$ for better gradients early in training.

## 3.4 Training Dynamics

**The discriminator loss:**

$$\mathcal{L}_D = -\mathbb{E}_{x \sim p_{data}}[\log D(x)] - \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

**The generator loss (alternative formulation):**

$$\mathcal{L}_G = -\mathbb{E}_{z \sim p_z}[\log D(G(z))]$$

## 3.5 Code: GAN (PyTorch)

```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, output_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

# Training loop
def train_gan(generator, discriminator, dataloader, epochs=100, latent_dim=100):
    criterion = nn.BCELoss()
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4)
    
    for epoch in range(epochs):
        for real_data, _ in dataloader:
            batch_size = real_data.size(0)
            
            # Train discriminator
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)
            
            # Real data
            d_real = discriminator(real_data)
            d_loss_real = criterion(d_real, real_labels)
            
            # Fake data
            z = torch.randn(batch_size, latent_dim)
            fake_data = generator(z)
            d_fake = discriminator(fake_data.detach())
            d_loss_fake = criterion(d_fake, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            
            # Train generator
            z = torch.randn(batch_size, latent_dim)
            fake_data = generator(z)
            g_output = discriminator(fake_data)
            g_loss = criterion(g_output, real_labels)  # Wants to fool discriminator
            
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
```

## 3.6 GAN Challenges

| Challenge | Problem | Mitigation |
|-----------|---------|------------|
| **Mode Collapse** | Generator produces limited variety | Wasserstein GAN, minibatch discrimination |
| **Training Instability** | Loss oscillates | Label smoothing, gradient penalties |
| **Vanishing Gradients** | Generator stops learning | Use log(D) formulation, WGAN |
| **Evaluation** | Hard to measure quality | FID, Inception Score |


---

# 4. DIFFUSION MODELS

## 4.1 What They Are

**Diffusion models** generate data by learning to reverse a gradual noising process. They start with pure noise and iteratively denoise it step by step to produce a sample.

**The core idea:** Instead of learning to generate in one step (like GANs), diffusion models learn the reverse of a Markov chain that gradually adds noise to data.

**Why this matters:** Diffusion models currently achieve the highest-quality image generation (DALL-E 2, Stable Diffusion, Sora) and are the dominant approach in generative AI.

## 4.2 The Architecture

**Forward Process (Noising):** Gradually add Gaussian noise to data over T steps:

$$x_t = \sqrt{1 - \beta_t} x_{t-1} + \sqrt{\beta_t} \epsilon_{t-1}$$

**Reverse Process (Denoising):** Learn to remove noise step by step:

$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

## 4.3 The Math

**Forward Process (Diffusion):**

$$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$$

**Efficient sampling (closed form):**

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon \quad \text{where } \alpha_t = 1-\beta_t, \bar{\alpha}_t = \prod_{i=1}^t \alpha_i$$

**The Training Objective (Simplified):**

$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \left[ ||\epsilon - \epsilon_\theta(x_t, t)||^2 \right]$$

The model learns to predict the noise $\epsilon$ that was added to the data.

**The reverse process sampling:**

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right) + \sigma_t z$$

## 4.4 Code: Diffusion Model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionModel:
    def __init__(self, model, T=1000, beta_start=1e-4, beta_end=0.02):
        self.model = model
        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # Precompute noise schedule
        self.betas = torch.linspace(beta_start, beta_end, T)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
    
    def forward_diffusion(self, x0, t):
        """Add noise to data at time t"""
        noise = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bars[t]
        sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar_t)
        xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
        return xt, noise
    
    def reverse_diffusion(self, shape, device):
        """Generate from noise"""
        x = torch.randn(shape, device=device)
        
        for t in reversed(range(self.T)):
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # Predict noise
            noise_pred = self.model(x, t_tensor)
            
            # Sample x_{t-1}
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]
            beta_t = self.betas[t]
            
            x = (1 / torch.sqrt(alpha_t)) * (
                x - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * noise_pred
            )
            
            # Add noise (except at t=0)
            if t > 0:
                noise = torch.randn_like(x)
                x = x + torch.sqrt(beta_t) * noise
        
        return x

# Simple UNet-like model for diffusion
class DiffusionUNet(nn.Module):
    def __init__(self, in_channels=3, time_dim=256):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Simplified for illustration
        self.encoder = nn.ModuleList([
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
        ])
        
        self.decoder = nn.ModuleList([
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.Conv2d(64, in_channels, 3, padding=1),
        ])
    
    def forward(self, x, t):
        # Embed time
        t_embed = self.time_embed(t)
        # Simplified forward pass
        for layer in self.encoder:
            x = F.silu(layer(x))
        for layer in self.decoder:
            x = F.silu(layer(x))
        return x
```

## 4.5 Diffusion vs Other Generators

| Aspect | GAN | VAE | Diffusion |
|--------|-----|-----|-----------|
| **Image Quality** | Good | Blurry | Excellent |
| **Training Stability** | Poor | Good | Good |
| **Mode Coverage** | Poor | Good | Excellent |
| **Sampling Speed** | Fast (1 step) | Fast (1 step) | Slow (1000 steps) |
| **Diversity** | Limited | Good | Excellent |
| **Training Difficulty** | Hard | Easy | Moderate |

## 4.6 What Makes Diffusion So Good

**Why diffusion beats GANs:** The iterative denoising process is more stable to train and covers the full data distribution better (no mode collapse). The objective is simple and well-behaved: predict the noise.

**Why diffusion beats VAEs:** The step-by-step generation allows for higher-quality outputs than VAEs' single-step reconstruction. The model doesn't need to compress all information into a single bottleneck.

**The tradeoff:** Diffusion is **slow** at inference (50-1000 steps). This is the focus of active research (Distilled Diffusion, Consistency Models).


---

# FULL COMPARISON TABLE

| Aspect | Autoencoder | VAE | GAN | Diffusion |
|--------|-------------|-----|-----|-----------|
| **Generative?** | No | Yes | Yes | Yes |
| **Latent Space** | Deterministic | Probabilistic | Random noise | No explicit latent |
| **Training** | Reconstruction only | Reconstruction + KL | Adversarial | Noise prediction |
| **Sample Quality** | N/A | Good | Excellent | Best |
| **Sample Diversity** | N/A | Good | Limited | Excellent |
| **Inference Speed** | Fast | Fast | Fast | Slow |
| **Training Stability** | High | High | Low | High |
| **Use Case** | Compression, Denoising | Generation | Fast generation | High-quality generation |


---

# QUICK DECISION RULES

1. **Need dimensionality reduction or denoising?** → Autoencoder
2. **Need to generate new samples with a smooth latent space?** → VAE
3. **Need fast, high-quality generation?** → GAN
4. **Need the highest quality, don't care about speed?** → Diffusion
5. **Need controllable generation with text prompts?** → Diffusion (Stable Diffusion)


---

# 5 MOST-ASKED GENERATIVE MODEL INTERVIEW QUESTIONS

**1. What's the difference between an autoencoder and a VAE?** A VAE adds probabilistic regularization to the latent space by imposing a prior distribution (typically Gaussian). This makes the latent space smooth and continuous, enabling generation by sampling from the prior.

**2. What is the ELBO in VAEs?** The Evidence Lower Bound is the objective function maximized in VAEs. It has two terms: reconstruction loss (how well the decoder reconstructs the input) and KL divergence (how close the learned distribution is to the prior).

**3. Why do GANs have mode collapse?** The discriminator and generator can get stuck in a local optimum where the generator finds a small set of samples that fool the discriminator. The generator then produces only these samples, collapsing to a single mode.

**4. Why do diffusion models produce higher quality than GANs?** Diffusion models learn to denoise step-by-step, which is a simpler, more stable training objective than adversarial training. They also avoid mode collapse because the denoising process covers the full data distribution.

**5. What's the reparameterization trick in VAEs?** To backpropagate through the random sampling operation, VAEs rewrite z = μ + σ⊙ε where ε ∼ 𝒩(0, I). This moves the randomness outside the gradient path, making the sampling differentiable.
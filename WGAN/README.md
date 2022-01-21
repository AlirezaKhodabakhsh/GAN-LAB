**Sources:**  
- [ ] https://jonathan-hui.medium.com/gan-gan-series-2d279f906e7b

# BCE
```python
def get_loss_dis(gen, dis,
                 real_image,
                 N_noise, C_noise,
                 device):

    x = real_image
    x_hat = gen( get_noise(N_noise, C_noise, device=device) ).detach()

    return torch.mean(-torch.log10(F.sigmoid(dis(x))), dim=0) + torch.mean(-torch.log10(1 - F.sigmoid(dis(x_hat))), dim=0 )
```

```python
def get_loss_gen(gen, dis,
                 N_noise, C_noise,
                 device):

    x_hat = gen( get_noise(N_noise, C_noise, device=device) )

    return torch.mean(torch.log10(1 - F.sigmoid(dis(x_hat))), dim=0 )
```

# WGAN 

# WGAN-GP
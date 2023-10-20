# Assignment 20 - Stable Diffusion Image Generation 

### The custom blue loss function 
Implemented contrast adjustment as error loss function to guide the latent space with selected contrast adjustment
```
def blue_loss(images, contrast_perc=80):
    # How far the pixels are from +80% contrast:
    contrast = 255*contrast_perc // 100 # it ranges from -255 to +255
    contrast_scale_factor = (259 * (contrast + 255)) / (255 * (259 - contrast))
    cimgs = (contrast_scale_factor * (images - 0.5) + 0.5 )
    cimgs = torch.where(cimgs > 1.0, 1.0, cimgs)
    cimgs = torch.where(cimgs < 0.0, 0.0, cimgs)
    error = torch.abs( images - cimgs ).mean()
    return error

```

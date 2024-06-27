from pathlib import Path
import tqdm

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, sampler
from torchvision import datasets, transforms, utils
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
import matplotlib.pyplot as plt
from Blocks.models import *

import gc


gc.collect()
torch.cuda.empty_cache()

def show(img):
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))


# We set a random seed to ensure that your results are reproducible.
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
torch.manual_seed(0)

GPU = True # Choose whether to use GPU
if GPU:
    device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(f'Using {device}')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#      # Inverse Transform (Unnormalize and Convert to PIL)
# inverse_transform = transforms.Compose([
#     transforms.Normalize((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0)),  # Unnormalize
#     transforms.ToPILImage()  # Convert back to PIL Image
# ])


# Necessary Hyperparameters
num_epochs = 100
learning_rate = 5e-5
batch_size = 64

# Additional Hyperparameters
hidden_dim = 64


trainset = datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')




# get some random training images
dataiter = iter(trainloader)
images, _ = next(dataiter)

# print(images.shape)
# print(images[0].min(), images[0].max())

# # show images
# show(make_grid(images))

# print labels
# print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

def mse(x, x_hat):
  '''
  x, x_hat (tensor)
  '''
  return ((x_hat-x)**2).mean()

def denorm(img):
    img = img / 2 + 0.5
    return img



# # Test actnorm layer
# actnorm = ActNorm(3,4,4)

# batch = torch.randn(1,3,4,4)
# print("### Actnorm layer test ###")
# with torch.no_grad():
#   a,_ = actnorm(batch)
#   b = actnorm.inverse(a)
#   print(torch.allclose(batch, b))
#   print(mse(batch, b))

model = Glow(n_flow_steps=32, n_flow_layers=3).to(device)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of parameters is: {}".format(params))
# print(model)
# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# print('### Glow Test ###')
# x = torch.randn(1, 3, 32, 32).to(device)
# with torch.no_grad():
#   z_0, log_det_jacobian_total = model(x)
#   x_hat = model.inverse(z_0)
#   print(mse(x,x_hat))
#   print(torch.allclose(x, x_hat))


def evaluate(model, title):
    # *CODE FOR PART 1.2b IN THIS CELL*

    # load the model
    print('Input images')
    print('-'*50)

    sample_inputs, _ = next(iter(testloader))
    fixed_input = sample_inputs[0:32, :, :, :]
    # visualize the original images of the last batch of the test set
    img = make_grid(denorm(fixed_input), nrow=8, padding=2, normalize=False,
                    value_range=None, scale_each=False, pad_value=0)

    save_image(img, f'results/{title}_source_image.png')
    # plt.figure()
    # show(img)

    print('Reconstructed images')
    print('-'*50)
    with torch.no_grad():
        # visualize the reconstructed images of the last batch of test set

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        recon_batch, _ = model.forward(fixed_input.to(device))
        recon_batch = model.inverse(recon_batch)
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


        recon_batch = recon_batch.cpu()
        recon_batch = make_grid(denorm(recon_batch), nrow=8, padding=2, normalize=False,
                                value_range=None, scale_each=False, pad_value=0)

        save_image(recon_batch, f'results/{title}_recon_image.png')
        # plt.figure()
        # show(recon_batch)

    print('Generated Images')
    print('-'*50)
    model.eval()
    n_samples = 16
    z = torch.randn(n_samples, 3, 32,32).to(device)
    with torch.no_grad():
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        samples = model.inverse(z)
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

        samples = samples.cpu()
        samples = make_grid(denorm(samples), nrow=8, padding=2, normalize=False,
                                value_range=None, scale_each=False, pad_value=0)
        save_image(samples, f'results/{title}_sampled_image.png')
        # plt.figure(figsize = (8,8))
        # show(samples)

evaluate(model, "before")

def test_layers(x, model):

    for layer in model.flow_layers:
      x = model.squeeze2d(x)
      for flow_step in layer:
        for module in flow_step.layers:
            z_i, _ = module(x)
            print(type(module).__name__, ":", mse(x, module.inverse(z_i)))
            print("-"*50)
            x = z_i


    return



torch.cuda.empty_cache()
def loss_function_Glow(target_distribution, z, log_det_jacobian):
  # print(log_det_jacobian)
  log_likelihood = target_distribution.log_prob(z).mean() + log_det_jacobian/z.shape[0]
  return -log_likelihood

from torch.distributions.normal import Normal

model.train()
target_distribution = Normal(torch.tensor(0, dtype=torch.float32).to(device),torch.tensor(1, dtype=torch.float32).to(device))
# <- You may wish to add logging info here
for epoch in range(num_epochs):
    total_loss = 0 # <- You may wish to add logging info here
    err = 0

    with tqdm.tqdm(trainloader, unit="batch") as tepoch:
        for batch_idx, (data, _) in enumerate(tepoch):
            #######################################################################
            #                       ** START OF YOUR CODE **
            #######################################################################
            data = data.to(device) # Need at least one batch/random data with right shape - .view(-1,28*28)
                        # This is required to initialize to model properly below
                        # when we save the computational graph for testing (jit.save)


            # forward pass
            z, log_det_jacobian = model.forward(data)

            # compute loss
            loss = loss_function_Glow(target_distribution, z, log_det_jacobian)


            # backwards
            optimizer.zero_grad()
            loss.backward()

            # update params
            optimizer.step()

            # Logging
            total_loss += loss.item()
            prior = target_distribution.log_prob(z).mean()
            # jacobian = log_det_jacobian/z.shape[0]
            





            # torch.cuda.empty_cache()
            #######################################################################
            #                       ** END OF YOUR CODE **
            #######################################################################
            if batch_idx % 20 == 0:
                with torch.no_grad():
                    err = mse(data, model.inverse(z)).mean()
                tepoch.set_description(f"Epoch {epoch}")
                tepoch.set_postfix(loss=loss.item()/len(data), log_prior=prior.item(), recon_err = err.item())

    # save the model
    if epoch == num_epochs - 1:
        with torch.no_grad():
            torch.jit.save(torch.jit.trace(model, (data), check_trace=False),
                'weights/Glow_model.pth')

evaluate(model, "after")


#sample_inputs, _ = next(iter(testloader))
#fixed_input = sample_inputs[0:32, :, :, :]
#test_layers(fixed_input.to(device), model)

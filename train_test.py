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
     #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])




# Necessary Hyperparameters
num_epochs = 70
learning_rate = 1e-4
batch_size = 16
n_bits = 5

# Additional Hyperparameters
# hidden_dim = 64
# max_clip = 0.02


trainset = datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)





# get some random training images
dataiter = iter(trainloader)
images, _ = next(dataiter)


def mse(x, x_hat):
  '''
  x, x_hat (tensor)
  '''
  return ((x_hat-x)**2).mean()





model = Glow(n_channels=3, n_steps=32, n_flow_blocks=3, n_bits=n_bits).to(device)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of parameters is: {}".format(params))
# print(model)
# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# def grad_size(model):
#     count = 0
#     average = 0
#     max_grad = 0
#     for w in model.parameters():
#         v = w.abs().mean() # number of elements in w
#         n = w.numel()
#         count += n
#         average += n * v
#         max_grad = max(w.abs().max(), max_grad )
#     return average / count , max_grad


    


def evaluate(model, title):
    # *CODE FOR PART 1.2b IN THIS CELL*
    

    # load the model
    print('Input images')
    print('-'*50)

    sample_inputs, _ = next(iter(testloader))
    fixed_input = sample_inputs[0:32, :, :, :]
    # visualize the original images of the last batch of the test set
    img = make_grid(fixed_input, nrow=8, padding=2, normalize=False,
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
        recon_batch, _ = model.forward(fixed_input.to(device), train=False)
        recon_batch = model.inverse(recon_batch)
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


        recon_batch = recon_batch.cpu()
        # if normalised is True:
        #     recon_batch = denorm(recon_batch)

        recon_batch = make_grid(recon_batch, nrow=8, padding=2, normalize=False,
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
        samples = model.inverse(model.z_to_list(z))
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

        samples = samples.cpu()
        # if normalised is True:
        #     samples = denorm(samples)
        samples = make_grid(samples, nrow=8, padding=2, normalize=False,
                                value_range=None, scale_each=False, pad_value=0)
        save_image(samples, f'results/{title}_sampled_image.png')
        # plt.figure(figsize = (8,8))
        # show(samples)

def rand_sample(model, tag, z):
    z = z.to(device)
    with torch.no_grad():
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        samples = model.inverse(model.z_to_list(z))
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

        samples = samples.cpu()
        samples = make_grid(samples, nrow=8, padding=2, normalize=False,
                                value_range=None, scale_each=False, pad_value=0)
        save_image(samples, f'results/sampled_at_epoch-{tag}.png')

evaluate(model, "before")


pi = torch.tensor(np.pi)
torch.cuda.empty_cache()
def loss_function_Glow(z_list, log_det_jacobian, n_bins=torch.tensor(256.)):
  # fit z to normal distribution


  log_p_z = 0
  for z_i in z_list:
    log_p_z += (-0.5 * (z_i ** 2 + torch.log(2 * pi ))).view(z_i.shape[0],-1).sum(-1)

  n_pixels = 3 * 32 * 32 # H * W * C HARDCODE FOR NOW
  c = -n_pixels * torch.log(n_bins)


  log_likelihood = log_p_z + log_det_jacobian + c
  loss = -log_likelihood

  loss_in_bits = (loss / (torch.log(torch.tensor(2.))  * n_pixels)).mean()
  log_p_z_in_bits = (log_p_z / (torch.log(torch.tensor(2.))  * n_pixels)).mean()
  jacobian_in_bits = (log_det_jacobian / (torch.log(torch.tensor(2.))  * n_pixels)).mean()

  return (
    loss_in_bits,
    log_p_z_in_bits, 
    jacobian_in_bits,
  )



model.train()
n_samples = 16
sample = torch.randn(n_samples, 3, 32, 32)
for epoch in range(num_epochs):
    total_loss = 0 # <- You may wish to add logging info here
    err = 0
    #max_grad = 0

    with tqdm.tqdm(trainloader, unit="batch") as tepoch:
        for batch_idx, (data, _) in enumerate(tepoch):
            #######################################################################
            #                       ** START OF YOUR CODE **
            #######################################################################
            data = data.to(device) # Need at least one batch/random data with right shape - .view(-1,28*28)
                        # This is required to initialize to model properly below
                        # when we save the computational graph for testing (jit.save)
            with torch.no_grad():
                _,_, = model.forward(data)


            # forward pass
            z_list, log_det_jacobian = model.forward(data)

            log_det_jacobian = log_det_jacobian.mean()

            # compute loss
            loss, prior, jacobian = loss_function_Glow( z_list, log_det_jacobian, n_bins=torch.tensor(2**n_bits))


            # backwards
            optimizer.zero_grad()
            loss.backward()

            # Clip gradient
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_clip)

            # update params
            optimizer.step()

            # Logging
            total_loss += loss.item()
            





            # torch.cuda.empty_cache()
            #######################################################################
            #                       ** END OF YOUR CODE **
            #######################################################################
            if batch_idx % 20 == 0:
                with torch.no_grad():
                    err = mse(data, model.inverse(z_list)).mean()
                tepoch.set_description(f"Epoch {epoch}")
                #avg_grad, max_grad = grad_size(model)
                tepoch.set_postfix(loss=loss.item()/len(data), log_prior=prior.item(), log_jacobian=jacobian.item(), recon_err = err.item() )#, avg_weights=avg_grad.item(), max_grad=max_grad.item() )

    if epoch % 1 == 0:
        rand_sample(model, epoch, sample)

    # save the model
    if epoch == num_epochs - 1:
        with torch.no_grad():
            torch.jit.save(torch.jit.trace(model, (data), check_trace=False),
                'weights/Glow_model.pth')

evaluate(model, "after")



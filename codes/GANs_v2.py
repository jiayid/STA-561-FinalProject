
# coding: utf-8

# In[5]:


import os
import argparse
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from Generator import generator
from Discriminator import discriminator
from Visualize import loss_plots,rotate
from tqdm import tqdm


# In[6]:


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

class generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=100, output_dim=1, input_size=32):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )
        initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)

        return x

class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=1, output_dim=1, input_size=32):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )
        initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc(x)

        return x


# In[113]:


transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
])


# In[7]:


def save_images(num_epoch, show = False, save = False, path = 'result.png',dataset_dir = 'EMNIST'):
    if (dataset_dir == 'MNIST'):
        dim = 4
    else:
        dim = 7

    z_ = torch.randn((dim*dim, 100))
    z_ = Variable(z_, volatile=True)

    G.eval()
    test_images = G(z_)
    G.train()

    size_figure_grid = dim
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(dim, dim))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)
    for k in range(dim*dim):
        i = k // dim
        j = k % dim
        ax[i, j].cla()
        ax[i, j].imshow((test_images[k, :].cpu().data.view(28, 28).numpy()), cmap='Greys_r')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.05, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


# In[8]:



################################### Main Code ######################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAN")
    parser.add_argument("--dataset_dir", type=str, default="MNIST",  ## directory is name of the dataset
                        help="which dataset")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="number of epochs to train (default: 100)")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="learning rate for training (default: 0.0002)")
    args = parser.parse_known_args()[0]
    dataset_dir = args.dataset_dir
    epochs = args.num_epochs
    lr = args.lr

    # creating folders for results
    if not os.path.isdir(dataset_dir):
        os.mkdir(dataset_dir)
    if not os.path.isdir(dataset_dir + '/images'):
        os.mkdir(dataset_dir+'/images')
    # data_loader
    # transforms.ToTensor() = torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    # Tensor image of size (C, H, W) to be normalized. i.e. input[channel] = (input[channel] - mean[channel]) / std[channel]

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=([0.5]), std=([0.5]))
    ])
    # if data not present then it downloads, takes train part
    print("loading dataset ...")
    if dataset_dir == 'EMNIST':
        train_loader = torch.utils.data.DataLoader(
            datasets.EMNIST(dataset_dir +'/data',split = 'bymerge', train=True, download=True, transform=transform),
            batch_size=128)

    if dataset_dir == 'MNIST':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(dataset_dir + '/data', train=True, download=True, transform=transform),
            batch_size=64,shuffle=True)


# In[99]:


# networks
G = generator(input_dim=100, output_dim=1, input_size=28)
D = discriminator(input_dim=1, output_dim=1, input_size=28)


# In[104]:



    # Adam optimizer
    G_optimizer = optim.Adam(G.parameters(), lr=lr)
    D_optimizer = optim.Adam(D.parameters(), lr=lr)

    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []
    for epoch in tqdm(range(50)):
        D_losses = []
        G_losses = []
        for load_data, _ in train_loader:

            # training discriminator ############
            # manually setting gradients to zero before mini batches 
            D.zero_grad()

            # format
#             load_data = load_data.view(-1, 28 * 28)
            # print load_data.size()[0]
            mini_batch = load_data.size()[0]

            D_real = torch.ones(mini_batch)
            D_fake = torch.zeros(mini_batch)

            # variables in pytorch can directly be accessed
#             load_data  = Variable(load_data.cuda())
#             D_real = Variable(D_real.cuda())
#             D_fake = Variable(D_fake.cuda())

            # first it takes real data 
            D_result = D(load_data)
            # loss calculations due to real data : first term in eqn
            # comparing with ones labels
            D_real_loss = F.binary_cross_entropy(D_result, D_real)
            # D_real_scores = D_result

            ## for loss due to generated samples
            noise = torch.randn((mini_batch, 100))
#             noise = Variable(noise.cuda())

            G_sample = G(noise)
            D_result = D(G_sample)
            # loss calculations due to generated data : second term in eqn
            # comparing with zero labels
            D_fake_loss = F.binary_cross_entropy(D_result, D_fake)
            # D_fake_scores = D_result
            # total D_loss
            D_train_loss = D_real_loss + D_fake_loss

            # training of network
            D_train_loss.backward()
            D_optimizer.step()

            D_losses.append(D_train_loss.data)

            # training generator ##############

            # manually setting gradients to zero before mini batches 
            G.zero_grad()

            noise = torch.randn((mini_batch, 100))
            out = torch.ones(mini_batch)

            # variables in pytorch can directly be accessed
#             noise = Variable(noise.cuda())
#             out = Variable(out.cuda())
            # noise input to generator 
            G_result = G(noise)
            D_result = D(G_result)
            # comparing with ones labels
            # loss calculations due to generated data : generator's loss
            G_train_loss = F.binary_cross_entropy(D_result, out)
            # training of network
            G_train_loss.backward()
            G_optimizer.step()

            G_losses.append(G_train_loss.data)

        print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
            (epoch + 1), epochs, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))

        p = dataset_dir + '/images/' + str(epoch + 1)+ '.png'
        save_images((epoch+1),  save=True, path=p, dataset_dir = dataset_dir)
        train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
        train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))



# In[3]:


# creating gif file     
images = []
for i in range(50):
    img_name = dataset_dir + '/images/' + str(i + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave(dataset_dir + '/gif_file.gif', images, fps=5)


# In[11]:



    print("Finished training!")

    ### showing and saving the results ###############
    loss_plots(train_hist, save=True, path=dataset_dir + '/EMNIST_GAN_train_hist.png')
    torch.save(G.state_dict(), dataset_dir + "/generator_param.pkl")
    torch.save(D.state_dict(), dataset_dir + "/discriminator_param.pkl")
    with open(dataset_dir + '/train_hist.pkl', 'wb') as f:
        pickle.dump(train_hist, f)

    # creating gif file     
    images = []
    for i in range(epochs):
        img_name = dataset_dir + '/images/' + str(i + 1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave(dataset_dir + '/gif_file.gif', images, fps=5)


# In[39]:


class generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=100, output_dim=1, input_size=32):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )
        initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)

        return x

class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=1, output_dim=1, input_size=32):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )
        initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc(x)

        return x


# In[43]:



    # networks
    G = generator(input_size=100, n_class=28*28)
    D = discriminator(input_size=28*28, n_class=1)
    # Adam optimizer
    G_optimizer = optim.Adam(G.parameters(), lr=lr)
    D_optimizer = optim.Adam(D.parameters(), lr=lr)

    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []
    for epoch in tqdm(range(epochs)):
        D_losses = []
        G_losses = []
        for load_data in train_loader:

            # training discriminator ############
            # manually setting gradients to zero before mini batches 
            D.zero_grad()

            # format
            load_data = load_data.view(-1, 28 * 28)
            # print load_data.size()[0]
            mini_batch = load_data.size()[0]

            D_real = torch.ones(mini_batch)
            D_fake = torch.zeros(mini_batch)

            # variables in pytorch can directly be accessed
            load_data  = Variable(load_data)
            D_real = Variable(D_real)
            D_fake = Variable(D_fake)

            # first it takes real data 
            D_result = D(load_data)
            # loss calculations due to real data : first term in eqn
            # comparing with ones labels
            D_real_loss = F.binary_cross_entropy(D_result, D_real)
            # D_real_scores = D_result

            ## for loss due to generated samples
            noise = torch.randn((mini_batch, 100))
            noise = Variable(noise)

            G_sample = G(noise)
            D_result = D(G_sample)
            # loss calculations due to generated data : second term in eqn
            # comparing with zero labels
            D_fake_loss = F.binary_cross_entropy(D_result, D_fake)
            # D_fake_scores = D_result
            # total D_loss
            D_train_loss = D_real_loss + D_fake_loss

            # training of network
            D_train_loss.backward()
            D_optimizer.step()

            D_losses.append(D_train_loss.data)

            # training generator ##############

            # manually setting gradients to zero before mini batches 
            G.zero_grad()

            noise = torch.randn((mini_batch, 100))
            out = torch.ones(mini_batch)

            # variables in pytorch can directly be accessed
            noise = Variable(noise)
            out = Variable(out)
            # noise input to generator 
            G_result = G(noise)
            D_result = D(G_result)
            # comparing with ones labels
            # loss calculations due to generated data : generator's loss
            G_train_loss = F.binary_cross_entropy(D_result, out)
            # training of network
            G_train_loss.backward()
            G_optimizer.step()

            G_losses.append(G_train_loss.data[0])

        print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
            (epoch + 1), epochs, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))

        p = dataset_dir + '/images/' + str(epoch + 1)+ '.png'
        save_images((epoch+1),  save=True, path=p, dataset_dir = dataset_dir)
        train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
        train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))


    print("Finished training!")


# In[ ]:



    ### showing and saving the results ###############
    loss_plots(train_hist, save=True, path=dataset_dir + '/EMNIST_GAN_train_hist.png')
    torch.save(G.state_dict(), dataset_dir + "/generator_param.pkl")
    torch.save(D.state_dict(), dataset_dir + "/discriminator_param.pkl")
    with open(dataset_dir + '/train_hist.pkl', 'wb') as f:
        pickle.dump(train_hist, f)

    # creating gif file     
    images = []
    for i in range(epochs):
        img_name = dataset_dir + '/images/' + str(i + 1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave(dataset_dir + '/gif_file.gif', images, fps=5)


# In[53]:


# networks
G = generator(input_size=28, output_dim=28*28)
D = discriminator(input_size=28, output_dim=1)

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr)
D_optimizer = optim.Adam(D.parameters(), lr=lr)

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
for epoch in tqdm(range(epochs)):
    D_losses = []
    G_losses = []
    batch_idx = np.random.choice(X.shape[0], size = n_batch)
    X_batch = X[batch_idx]
    train_loader = X_batch
    train_loader = transform(train_loader)
    for load_data in train_loader:

        # training discriminator ############
        # manually setting gradients to zero before mini batches 
#             D.zero_grad()

        # format
        load_data = load_data.view(-1, 28 * 28)
        # print load_data.size()[0]
        mini_batch = load_data.size()[0]

        D_real = torch.ones(mini_batch)
        D_fake = torch.zeros(mini_batch)

        # variables in pytorch can directly be accessed
        load_data  = Variable(load_data)
        D_real = Variable(D_real)
        D_fake = Variable(D_fake)

        # first it takes real data 
        D_result = D(load_data)
        # loss calculations due to real data : first term in eqn
        # comparing with ones labels
        D_real_loss = F.binary_cross_entropy(D_result, D_real)
        # D_real_scores = D_result

        ## for loss due to generated samples
        noise = torch.randn((mini_batch, 100))
        noise = Variable(noise)

        G_sample = G(noise)
        D_result = D(G_sample)
        # loss calculations due to generated data : second term in eqn
        # comparing with zero labels
        D_fake_loss = F.binary_cross_entropy(D_result, D_fake)
        # D_fake_scores = D_result
        # total D_loss
        D_train_loss = D_real_loss + D_fake_loss

        # training of network
        D_train_loss.backward()
        D_optimizer.step()

        D_losses.append(D_train_loss.data)

        # training generator ##############

        # manually setting gradients to zero before mini batches 
        G.zero_grad()

        noise = torch.randn((mini_batch, 100))
        out = torch.ones(mini_batch)

        # variables in pytorch can directly be accessed
        noise = Variable(noise)
        out = Variable(out)
        # noise input to generator 
        G_result = G(noise)
        D_result = D(G_result)
        # comparing with ones labels
        # loss calculations due to generated data : generator's loss
        G_train_loss = F.binary_cross_entropy(D_result, out)
        # training of network
        G_train_loss.backward()
        G_optimizer.step()

        G_losses.append(G_train_loss.data)

#         print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
#             (epoch + 1), epochs, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))

    p = dataset_dir + '/images/' + str(epoch + 1)+ '.png'
    save_images((epoch+1), save=True, path=p, dataset_dir = dataset_dir)
    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))


print("Finished training!")


# In[42]:


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


# In[65]:


def save_images(num_epoch, show = False, save = False, path = 'result.png',dataset_dir = 'EMNIST'):
    if (dataset_dir == 'MNIST'):
        dim = 4
    else:
        dim = 7

    z_ = torch.randn((dim*dim, 100))

    G.eval()
    test_images = G(z_)
    G.train()

    size_figure_grid = dim
    
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(dim, dim))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)
    for k in range(dim*dim):
        i = k // dim
        j = k % dim
        ax[i, j].cla()
        ax[i, j].imshow(rotate(test_images[k, :].cpu().data.view(28, 28).numpy()), cmap='Greys_r')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.05, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


# In[117]:


### showing and saving the results ###############
loss_plots(train_hist, save=True, path=dataset_dir + '/EMNIST_GAN_train_hist.png')
torch.save(G.state_dict(), dataset_dir + "/generator_param.pkl")
torch.save(D.state_dict(), dataset_dir + "/discriminator_param.pkl")
with open(dataset_dir + '/train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

# creating gif file     
images = []
for i in range(epochs):
    img_name = dataset_dir + '/images/' + str(i + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave(dataset_dir + '/gif_file.gif', images, fps=5)


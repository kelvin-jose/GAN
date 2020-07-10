import config
import torch
import numpy


def train_generator(g_model, d_model, device, g_optimizer, loss_fn):
    images = torch.randn(config.BATCH_SIZE, config.LATENT_SIZE).to(device)
    labels = torch.ones(config.BATCH_SIZE, 1).to(device)

    output = g_model(images)
    loss = loss_fn(d_model(output), labels)

    g_optimizer.zero_grad()
    loss.backward()
    return loss


def train_discriminator(images, d_model, g_model, d_optimizer, device, loss_fn):
    d_optimizer.zero_grad()

    labels = torch.ones(config.BATCH_SIZE, 1).to(device)
    output_0 = d_model(images.reshape(config.BATCH_SIZE, -1))
    loss_0 = loss_fn(output_0, labels)

    fake_images = torch.randn(config.BATCH_SIZE, config.LATENT_SIZE).to(device)
    fake_labels = torch.zeros(config.BATCH_SIZE, 1).to(device)
    output_1 = g_model(fake_images)
    loss_1 = loss_fn(d_model(output_1), fake_labels)

    total_loss = loss_0 + loss_1
    total_loss.backward()

    d_optimizer.step()
    return total_loss, loss_0, loss_1


def train(data_loader, d_model, g_model, d_optimizer, g_optimizer, device, loss_fn):
    _total_loss, _real_loss, _fake_loss, _gen_loss = [], [], [], []
    for ix in range(config.EPOCHS):
        t_loss, r_loss, f_loss, g_loss = [], [], [], []
        for images, _ in data_loader:
            total_loss, real_loss, fake_loss = train_discriminator(images, d_model, g_model, d_optimizer, device,
                                                                   loss_fn)
            gen_loss = train_generator(g_model, d_model, device, g_optimizer, loss_fn)
            t_loss.append(total_loss)
            r_loss.append(real_loss)
            f_loss.append(fake_loss)
            g_loss.append(gen_loss)
        _total_loss.append(numpy.mean(t_loss))
        _real_loss.append(numpy.mean(r_loss))
        _fake_loss.append(numpy.mean(f_loss))
        _gen_loss.append(numpy.mean(g_loss))

    return _total_loss, _real_loss, _fake_loss, _gen_loss

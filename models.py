import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Function


class ProtoNet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

        if args.model_type == "ResNet":
            self.encoder = models.resnet50(pretrained=False)
            self.dims = [1000, 512, 256, 128]
        else:
            raise ValueError('')

        if self.args.pretrained:
            self.load()

        if args.model == 'qproto':
            self.vqvae = VectorQuantizedVAE(dims=self.dims, dim=self.dims[-1], K=args.k)

    def load(self):
        pre_trined = ResNet()
        pre_trined.load_state_dict(torch.load(self.args.weight_path))
        self.encoder.load_state_dict(pre_trined.encoder.state_dict())
        del pre_trined

    def forward(self, data_shot, data_query):
        feautures = self.encoder(data_shot)

        if self.args.model == 'qproto':
            x_tilde, z_e_x, z_q_x = self.vqvae(feautures)
            proto = z_e_x.reshape(self.args.shot, self.args.way, -1).mean(dim=0)
            data_query = self.encoder(data_query)
            data_query = self.vqvae.encoder(data_query)
            logits = self.euclidean_metric(data_query, proto) / self.args.temperature
            return logits, feautures, x_tilde, z_e_x, z_q_x

        elif self.args.model == 'proto':
            proto = feautures.reshape(self.args.shot, self.args.way, -1).mean(dim=0)
            logits = self.euclidean_metric(self.encoder(data_query), proto) / self.args.temperature
            return logits

    @staticmethod
    def euclidean_metric(a, b):
        n = a.shape[0]
        m = b.shape[0]
        a = a.unsqueeze(1).expand(n, m, -1)
        b = b.unsqueeze(0).expand(n, m, -1)
        logits = -((a - b) ** 2).sum(dim=2)
        return logits


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()

class VectorQuantization(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            inputs_flatten = inputs.view(-1, embedding_size)

            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                                    inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

            _, indices_flatten = torch.min(distances, dim=1)
            indices = indices_flatten.view(*inputs_size[:-1])
            ctx.mark_non_differentiable(indices)

            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
                           '`VectorQuantization`. The function `VectorQuantization` '
                           'is not differentiable. Use `VectorQuantizationStraightThrough` '
                           'if you want a straight-through estimator of the gradient.')


class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        indices = vq(inputs, codebook)
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = torch.index_select(codebook, dim=0,
                                           index=indices_flatten)
        codes = codes_flatten.view_as(inputs)

        return (codes, indices_flatten)

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)

            grad_output_flatten = (grad_output.contiguous()
                                   .view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return (grad_inputs, grad_codebook)

vq = VectorQuantization.apply
vq_st = VectorQuantizationStraightThrough.apply

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


class ResNet(nn.Module):

    def __init__(self, num_classes=100):
        super().__init__()
        self.encoder = models.resnet50(pretrained=False)
        self.fc = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x


class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1. / K, 1. / K)

    def forward(self, z_e_x):
        #         z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        latents = vq(z_e_x, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x):
        z_e_x_ = z_e_x
        #         z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
        #         z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()
        z_q_x = z_q_x_
        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
                                               dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        #         z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()
        z_q_x_bar = z_q_x_bar_
        return z_q_x, z_q_x_bar


class VectorQuantizedVAE(nn.Module):
    def __init__(self, dims=[1000, 512, 256, 128], dim=128, K=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.Linear(dims[1], dims[2]),
            nn.ReLU(True),
            nn.Linear(dims[2], dims[3])
        )

        self.codebook = VQEmbedding(K, dim)

        self.decoder = nn.Sequential(
            nn.Linear(dims[3], dims[2]),
            nn.ReLU(True),
            nn.Linear(dims[2], dims[1]),
            nn.Linear(dims[1], dims[0])
        )

        self.apply(weights_init)

    def encode(self, x):
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return latents

    def decode(self, latents):
        z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)
        x_tilde = self.decoder(z_q_x)
        return x_tilde

    def forward(self, x):
        z_e_x = self.encoder(x)

        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)

        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x



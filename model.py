import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATConv, global_mean_pool


# =========================================================
# 1. GVAE ENCODER
# =========================================================
class GVAE_Encoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim):
        super(GVAE_Encoder, self).__init__()

        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv_mu = GCNConv(hidden_dim, latent_dim)
        self.conv_logvar = GCNConv(hidden_dim, latent_dim)

    def forward(self, x, edge_index):

        h = F.relu(self.conv1(x, edge_index))

        mu = self.conv_mu(h, edge_index)
        logvar = self.conv_logvar(h, edge_index)

        return mu, logvar


# =========================================================
# 2. REPARAMETERIZATION TRICK
# =========================================================
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


# =========================================================
# 3. GVAE DECODER (INNER PRODUCT)
# =========================================================
class GVAE_Decoder(nn.Module):
    def forward(self, z, edge_index):
        src, dst = edge_index
        return torch.sigmoid((z[src] * z[dst]).sum(dim=1))


# =========================================================
# 4. MESSAGE PASSING NETWORK (MPNN CLASSIFIER)
# =========================================================
class MPNN(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_layers=3):
        super(MPNN, self).__init__()

        self.num_layers = num_layers

        self.convs = nn.ModuleList([
            GATConv(latent_dim if i == 0 else hidden_dim,
                    hidden_dim)
            for i in range(num_layers)
        ])

        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, batch):

        h = x

        for conv in self.convs:
            m = F.relu(conv(h, edge_index))
            h = self.gru(m, h)

        # graph-level representation
        g = global_mean_pool(h, batch)

        out = torch.sigmoid(self.classifier(g))
        return out


# =========================================================
# 5. FULL VMPNN MODEL (GVAE + MPNN)
# =========================================================
class VMPNN(nn.Module):
    def __init__(self,
                 node_feature_dim,
                 hidden_dim=64,
                 latent_dim=32,
                 num_layers=3):

        super(VMPNN, self).__init__()

        # GVAE
        self.encoder = GVAE_Encoder(node_feature_dim,
                                     hidden_dim,
                                     latent_dim)

        self.decoder = GVAE_Decoder()

        # MPNN classifier
        self.mpnn = MPNN(latent_dim,
                         hidden_dim,
                         num_layers)

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch

        # =========================
        # GVAE ENCODING
        # =========================
        mu, logvar = self.encoder(x, edge_index)

        z = reparameterize(mu, logvar)

        # =========================
        # GVAE RECONSTRUCTION LOSS
        # =========================
        recon_edge_prob = self.decoder(z, edge_index)

        # =========================
        # MPNN CLASSIFICATION
        # =========================
        sat_pred = self.mpnn(z, edge_index, batch)

        return sat_pred, mu, logvar, recon_edge_prob


# =========================================================
# 6. LOSS FUNCTION (GVAE + CLASSIFIER)
# =========================================================
def loss_function(pred, target, mu, logvar, recon, edge_label):

    # SAT classification loss
    cls_loss = F.binary_cross_entropy(pred.view(-1), target.float())

    # Reconstruction loss
    recon_loss = F.binary_cross_entropy(recon, edge_label.float())

    # KL divergence
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return cls_loss + recon_loss + kl_loss

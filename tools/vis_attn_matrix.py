import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from matplotlib.gridspec import GridSpec
import os
import sys

def calculate_attn_map(params_for_debug_per_example, device='cuda:0', average_channels=True):
    if average_channels:
        L = params_for_debug_per_example['delta_t'].shape[2]
        D = params_for_debug_per_example['delta_t'].shape[1]
        attn_map = torch.zeros(1, L, L, device=device)
        selected_channels = range(0, D)
    else:
        attn_map = []
        selected_channels = range(0, params_for_debug_per_example['A'].shape[0], 50)

    A = params_for_debug_per_example['A'].to(device)
    delta_t = params_for_debug_per_example['delta_t'][0].squeeze().to(device)
    Sb_x = params_for_debug_per_example['Sb_x'][0].squeeze().to(device)
    C_t = params_for_debug_per_example['C'][0].squeeze().to(device)

    if Sb_x.dim() != 2:
        Sb_x = Sb_x.unsqueeze(0)

    B_t = torch.einsum('ik,jk->ijk', delta_t, Sb_x)

    L = delta_t.shape[1]
    ND = len(selected_channels)
    d_state_space = C_t.shape[0]

    if average_channels:
        for c, c_idx in enumerate(tqdm(selected_channels)):
            delta_sum_map = torch.zeros(L, L, device=device)
            for i in range(1, L):
                delta_sum_map[i, 0:i] = delta_t[c_idx, i]
            delta_sum_map = torch.cumsum(delta_sum_map, dim=0)

            A_delta_t = torch.kron(delta_sum_map, A[c_idx, :].unsqueeze(dim=1).to(device))
            B_t_expanded = B_t[c_idx].repeat(L, 1)
            A_t_B_t = torch.exp(A_delta_t) * B_t_expanded
            C_t_expanded = torch.block_diag(*C_t.T).to(device)
            attn_map[0] += torch.tril(C_t_expanded @ A_t_B_t) / len(selected_channels)
    else:
        delta_sum_map = torch.zeros(ND, L, L, device=device)
        for i in range(1, L):
            for c, c_idx in enumerate(selected_channels):
                delta_sum_map[c, i, 0:i] = delta_t[c_idx, i]
        delta_sum_map = torch.cumsum(delta_sum_map, dim=1)

        A_delta_t = torch.cat(
            [torch.kron(delta_sum_map[c], A[c_idx, :].unsqueeze(dim=1).to(device)).unsqueeze(dim=0) for c, c_idx in
             enumerate(selected_channels)], dim=0)
        B_t_expanded = torch.cat(
            [B_t[c_idx].repeat(L, 1).unsqueeze(dim=0) for c, c_idx in enumerate(selected_channels)], dim=0)
        A_t_B_t = torch.exp(A_delta_t) * B_t_expanded
        C_t_expanded = torch.block_diag(*C_t.T).to(device)

        attn_map.append(torch.cat(
            [(torch.tril(C_t_expanded @ A_t_B_t[c])).unsqueeze(dim=0) for c, c_idx in enumerate(selected_channels)],
            dim=0))

    if not average_channels:
        attn_map = torch.cat(attn_map, dim=0)

    attn_map_normalized = torch.abs(attn_map.clone())
    attn_map_normalized[0, :, :] = attn_map_normalized[0, :, :] / torch.max(attn_map_normalized[0, :, :])

    return attn_map_normalized


def visualize_attn_map(attn_map_normalized, filename):
    TH_power = -2
    attn_map_db = torch.max(
        TH_power * torch.ones(attn_map_normalized[0].shape, device=device),
        torch.log10(torch.abs(attn_map_normalized[0]) ** 0.6).to(torch.float32)
    )
    show_len = 32
    attn_map_db_64 = attn_map_db[attn_map_db.shape[0] // 2: attn_map_db.shape[0] // 2 + show_len,
                     attn_map_db.shape[1] // 2: attn_map_db.shape[1] // 2 + show_len]

    fig, ax = plt.subplots()
    im = ax.imshow(attn_map_db.cpu(), interpolation='none', aspect='auto', cmap='turbo')
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(f'{filename}.pdf', format='pdf')
    plt.show()


def main(pt_files, device='cuda:0', average_channels=True):
    for pt_file in pt_files:
        filename = os.path.splitext(os.path.basename(pt_file))[0]
        params_for_debug_per_example = torch.load(pt_file, map_location='cpu')
        attn_map_normalized = calculate_attn_map(params_for_debug_per_example, device, average_channels)
        visualize_attn_map(attn_map_normalized, f'attn_matrix_{filename}')


if __name__ == "__main__":
    pt_files = sys.argv[1:]
    device = 'cuda:0'
    average_channels = True
    if len(pt_files) == 0:
        print("Please provide at least one .pt file for visualization.")
    else:
        main(pt_files, device, average_channels)

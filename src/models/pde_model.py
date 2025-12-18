import torch
import torch.nn as nn
import numpy as np

class StressPDE(nn.Module):
    def __init__(self, n_users, adj_matrix, ude_model, diffusion_coeff=0.1):
        super(StressPDE, self).__init__()
        self.n_users = n_users
        self.ude_model = ude_model
        self.diffusion_coeff = diffusion_coeff
        
        # Compute Graph Laplacian: L = D - A
        # A is adjacency (n, n)
        # D is degree matrix (diagonal)
        A = torch.tensor(adj_matrix, dtype=torch.float32)
        D = torch.diag(torch.sum(A, dim=1))
        L = D - A
        
        # Register L as a buffer (not a learnable parameter for now, unless we want to learn the graph)
        self.register_buffer('L', L)
        
    def forward(self, t, S_group):
        """
        t: scalar time
        S_group: (n_users, 1) stress values for all users at time t
        """
        # 1. Individual Dynamics (Reaction)
        reaction = self.ude_model(t, S_group) # (n_users, 1)
        
        # 2. Group Dynamics (Diffusion with Biased Kernel)
        # Standard Diffusion: -L * S  (where L = D - A)
        # This is equivalent to sum_j A_ij * (S_j - S_i)
        
        # We want to implement a "Biased Kernel" K(S_i, S_j)
        # Equation: dS_i/dt = ... + gamma * sum_j A_ij * K(S_i, S_j) * (S_j - S_i)
        
        # Efficient Vectorized Implementation:
        # Compute pairwise differences matrix: Diff_ij = S_j - S_i
        # S_group is (N, 1)
        # Diff matrix (N, N): Row i contains (S_0-S_i, S_1-S_i, ...)
        S_diff = S_group.T - S_group # (1, N) - (N, 1) -> (N, N). Entry (i, j) is S_j - S_i
        
        # Kernel Function K(diff):
        # Example: Gaussian Kernel (Social Influence decays with "stress distance")
        # K(x) = exp(-x^2 / sigma^2)
        # If sigma is small, people only influence those with VERY similar stress levels (Echo chambers).
        # If sigma is large, it approaches standard diffusion.
        sigma = 1.5  # Increased from 0.5 for stronger group influence
        K = torch.exp(- (S_diff**2) / (sigma**2))
        
        # Apply Adjacency Mask (only connected neighbors influence)
        # We use the adjacency matrix A stored in self.L (L = D - A, so A = D - L... or just pass A)
        # Actually, we only stored L. Let's recover A or just rely on the fact that we need A.
        # To be clean, let's re-store A in __init__.
        # For now, let's assume standard diffusion if we can't easily get A, 
        # BUT we can reconstruct A from L if D is diagonal. 
        # L_ij = -A_ij for i != j. So A_ij = -L_ij.
        
        A = -self.L.clone()
        A.fill_diagonal_(0) # A has 0 on diagonal
        
        # Weighted Influence
        # Influence_ij = A_ij * K_ij * (S_j - S_i)
        Influence = A * K * S_diff
        
        # Sum over j for each i
        diffusion = self.diffusion_coeff * torch.sum(Influence, dim=1, keepdim=True)
        
        return reaction + diffusion

def create_grid_graph(n_rows, n_cols):
    """
    Create a grid graph adjacency matrix.
    """
    n = n_rows * n_cols
    adj = np.zeros((n, n))
    
    for r in range(n_rows):
        for c in range(n_cols):
            i = r * n_cols + c
            # Neighbors: up, down, left, right
            if r > 0: adj[i, (r-1)*n_cols + c] = 1
            if r < n_rows - 1: adj[i, (r+1)*n_cols + c] = 1
            if c > 0: adj[i, r*n_cols + (c-1)] = 1
            if c < n_cols - 1: adj[i, r*n_cols + (c+1)] = 1
            
    return adj

def create_similarity_graph(user_features, k=5, threshold=0.5):
    """
    Create a graph based on feature similarity (e.g., baseline stress, resilience).
    Matches the 'Cohort Grouping' step in the architecture.
    
    user_features: (n_users, n_features) numpy array
    k: number of nearest neighbors
    """
    from sklearn.neighbors import kneighbors_graph
    
    # KNN graph
    A = kneighbors_graph(user_features, n_neighbors=k, mode='connectivity', include_self=False)
    adj = A.toarray()
    
    # Make symmetric (undirected graph for diffusion)
    adj = np.maximum(adj, adj.T)
    
    return adj

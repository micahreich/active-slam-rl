import torch
import normflows as nf


class SqueezeTemporal(nf.flows.Flow):
    """
    Squeeze operation for temporal (1D) data.
    
    Forward:
      Input:  (B, C, T)  with C divisible by 2.
      Process:
        - Reshape to (B, C//2, 2, T)
        - Permute to (B, C//2, T, 2)
        - Reshape to (B, C//2, 2T)
      Output: (B, C//2, 2T)
    
    Inverse:
      Input:  (B, C, T)  where T is even (and here C = original C//2).
      Process:
        - Reshape to (B, C, T//2, 2)
        - Permute to (B, C, 2, T//2)
        - Reshape to (B, 2C, T//2)
      Output: (B, 2C, T//2)
      
    Note: The log-determinant for a squeeze/unsqueeze operation is zero.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, z, context=None):
        """
        Forward squeeze.
        
        Args:
            z (Tensor): Input tensor of shape (B, C, T) with C divisible by 2.
            
        Returns:
            z (Tensor): Squeezed tensor of shape (B, C//2, 2T).
            log_det (float): Zero (squeeze is volume preserving).
        """
        log_det = 0
        B, C, T = z.size()
        # Reshape to split the channel dimension into two parts:
        # (B, C, T) -> (B, C//2, 2, T)
        z = z.view(B, C // 2, 2, T)
        # Permute to bring the factor of 2 into the time dimension:
        # (B, C//2, 2, T) -> (B, C//2, T, 2)
        z = z.permute(0, 1, 3, 2).contiguous()
        # Merge the last two dimensions:
        # (B, C//2, T, 2) -> (B, C//2, 2T)
        z = z.view(B, C // 2, T * 2)
        return z, log_det

    def inverse(self, z, context=None):
        """
        Inverse squeeze (unsqueeze).
        
        Args:
            z (Tensor): Input tensor of shape (B, C, T) where T is even.
                        Here C corresponds to the squeezed channels (original C//2).
                        
        Returns:
            z (Tensor): Unsqueezed tensor of shape (B, 2C, T//2).
            log_det (float): Zero (unsqueeze is volume preserving).
        """
        log_det = 0
        B, C, T = z.size()  # Here, T is assumed to be even (T = 2 * original T)
        # Reshape to separate the temporal factor:
        # (B, C, T) -> (B, C, T//2, 2)
        z = z.view(B, C, T // 2, 2)
        # Permute to move the factor into the channel dimension:
        # (B, C, T//2, 2) -> (B, C, 2, T//2)
        z = z.permute(0, 1, 3, 2).contiguous()
        # Merge the channel factor:
        # (B, C, 2, T//2) -> (B, 2C, T//2)
        z = z.view(B, C * 2, T // 2)
        return z, log_det
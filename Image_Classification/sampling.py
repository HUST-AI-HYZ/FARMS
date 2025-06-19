import math
import torch
import logging
logger = logging.getLogger(__name__)
UNKNOWN = 'unknown'

########################################################################################################
############################ This Code handles slicing of Conv2d layers ################################
########################################################################################################
class CHANNELS():
    UNKNOWN = UNKNOWN
    FIRST = 'first'
    LAST = 'last'

def channel_str(channel):
        if channel == CHANNELS.FIRST:
            return "FIRST"
        elif channel == CHANNELS.LAST:
            return "LAST"
        else:
            return "UNKNOWN"

def conv2D_Wmats(Wtensor, channels=CHANNELS.LAST):
        """Extract W slices from a 4 layer_id conv2D tensor of shape: (N,M,i,j) or (M,N,i,j).  
        Return ij (N x M) matrices, with receptive field size (rf) and channels flag (first or last)"""
        
        logger.debug("conv2D_Wmats")
        
        # TODO:  detect or use CHANNELS
        # if channels specified ...
    
        Wmats = []
        s = Wtensor.shape
        N, M, imax, jmax = s[0], s[1], s[2], s[3]
        
        if N + M >= imax + jmax:
            detected_channels= CHANNELS.LAST
        else:
            detected_channels= CHANNELS.FIRST
            

        if channels == CHANNELS.UNKNOWN :
            logger.debug(f"channels UNKNOWN, detected {channel_str(detected_channels)}")
            channels= detected_channels

        if detected_channels == channels:
            if channels == CHANNELS.LAST:
                logger.debug("channels Last tensor shape: {}x{} (NxM), {}x{} (i,j)".format(N, M, imax, jmax))                
                for i in range(imax):
                    for j in range(jmax):
                        W = Wtensor[:, :, i, j]
                        if W.shape[0] < W.shape[1]:
                            N, M = M, N
                            W = W.T
                        Wmats.append(W)
                        
            else: #channels == CHANNELS.FIRST  # i, j, M, N
                M, N, imax, jmax = imax, jmax, N, M
                # check this       
                logger.debug("channels First shape: {}x{} (NxM), {}x{} (i,j)".format(N, M, imax, jmax))                
                for i in range(imax):
                    for j in range(jmax):
                        W = Wtensor[i, j, :, :]
                        if W.shape[1] < W.shape[0]:
                            N, M = M, N
                            W = W.T
                        Wmats.append(W)
                            
        elif detected_channels != channels:
            logger.warning("warning, expected channels {},  detected channels {}".format(channel_str(channels),channel_str(detected_channels)))
            # flip how we extract the Wmats
            # reverse of above extraction
            if detected_channels == CHANNELS.LAST:
                logger.debug("Flipping LAST to FIRST Channel, {}x{} ()x{}".format(N, M, imax, jmax))   
                for i in range(N):
                    for j in range(M):
                        W = Wtensor[i, j,:,:]
                        if imax < jmax:
                            W = W.T
                        Wmats.append(W)
                        
            else: #detected_channels == CHANNELS.FIRST:
                N, M, imax, jmax = imax, jmax, N, M   
                logger.debug("Flipping FIRST to LAST Channel, {}x{} ()x{}".format(N, M, imax, jmax))                
                # check this       
                for i in range(N):
                    for j in range(M):
                        W = Wtensor[:, :, i, j]
                        if imax < jmax:
                            W = W.T
                        Wmats.append(W)
            # final flip            
            N, M, imax, jmax = imax, jmax, N, M   
           
                
        rf = imax * jmax  # receptive field size             
        logger.debug("get_conv2D_Wmats N={} M={} rf= {} channels= {}".format(N, M, rf, channels))
    
        return Wmats, N, M, rf



########################################################################################################
# This function applies sliding window sampling matrix method for approximating random matrix ESD (V1) #
########################################################################################################
def matrix_size_dependent_sampling(matrix, isconv2d, conv_norm, num_row_samples, Q_ratio, step_size=10):
    rows, cols = matrix.shape
    num_col_samples = int(num_row_samples * Q_ratio)
    
    # Calculate number of windows
    num_row_windows = max(1, (rows - num_row_samples) // step_size + 1)
    num_col_windows = max(1, (cols - num_col_samples) // step_size + 1)
    
    all_eigs = []
    # print(f"Matrix shape: ({rows}, {cols}) | Number of sliding operations across rows: {num_row_windows} and across columns: {num_col_windows}")
    
    for i in range(0, num_row_windows):
        for j in range(0, num_col_windows):
            # Extract submatrix
            row_start = i * step_size
            col_start = j * step_size
            submatrix = matrix[row_start:row_start + num_row_samples, 
                               col_start:col_start + num_col_samples]
            if isconv2d:
                submatrix *= math.sqrt(conv_norm)
            eigs = torch.square(torch.linalg.svdvals(submatrix))
            all_eigs.append(eigs)

    all_eigs = torch.cat(all_eigs)
    all_eigs, _ = torch.sort(all_eigs, descending=False)

    return all_eigs

########################################################################################################
# This function applies sliding window sampling matrix method for approximating random matrix ESD (V2) #
########################################################################################################
def fixed_sampling(matrix, isconv2d, conv_norm, num_row_samples, Q_ratio, sampling_ops_per_dim):
    rows, cols = matrix.shape
    num_col_samples = int(num_row_samples * Q_ratio)
    
    # Calculate step sizes
    row_step = max(1, (rows - num_row_samples) // (sampling_ops_per_dim - 1))
    col_step = max(1, (cols - num_col_samples) // (sampling_ops_per_dim - 1))
    
    # Calculate number of windows
    num_row_windows = min(sampling_ops_per_dim, max(1, (rows - num_row_samples) // row_step + 1))
    num_col_windows = min(sampling_ops_per_dim, max(1, (cols - num_col_samples) // col_step + 1))
    
    all_eigs = []
    # print(f"Matrix shape: ({rows}, {cols}) | Number of sliding operations across rows: {num_row_windows} and across columns: {num_col_windows}")
    # print(f"Row step size: {row_step}, Column step size: {col_step}")
    
    for i in range(0, num_row_windows):
        for j in range(0, num_col_windows):
            # Extract submatrix
            row_start = min(i * row_step, rows - num_row_samples)
            col_start = min(j * col_step, cols - num_col_samples)
            submatrix = matrix[row_start:row_start+num_row_samples, 
                               col_start:col_start+num_col_samples]
            if isconv2d:
                submatrix *= math.sqrt(conv_norm)
            eigs = torch.square(torch.linalg.svdvals(submatrix))
            all_eigs.append(eigs)
    
    all_eigs = torch.cat(all_eigs)
    all_eigs, _ = torch.sort(all_eigs, descending=False)

    return all_eigs





########################################################################################################
###################################### Driver Function #################################################
########################################################################################################

def sampled_eigs(matrix, isconv2d, conv_norm, num_row_samples, Q_ratio, sampling_ops_per_dim, step_size=10):
    # Using Conv2d weight slicing method from WW
    # if isconv2d:
        # matrix = matrix.view(matrix.size(0), -1) # * math.sqrt(conv_norm)
    # Use the conv2D_Wmats method to slice the Conv2D weight tensor
    # Wmats, N, M, rf = conv2D_Wmats(matrix, channels=CHANNELS.UNKNOWN)     
    
    #if M > N:
    #    Q_ratio = 2 - Q_ratio
        
    if isconv2d:
        raise NotImplementedError("Conv2D layer sampling is not implemented yet.")
        #eigs = sample_conv2d(matrix, conv_norm, Q_ratio)
        s = matrix.shape
        N, M, imax, jmax = s[0], s[1], s[2], s[3]
        
        
        eigs = sample_conv2d(matrix, conv_norm, Q_ratio, num_row_samples, sampling_ops_per_dim)
        print(f"Conv2D shape: {N}, {M}, {imax}, {jmax}. Length of eigs: {len(eigs)}")

        return eigs
        # Alternatively, use the following code to flatten the Conv2D weight tensor
        # matrix = matrix.view(matrix.size(0), -1) * math.sqrt(conv_norm)
    
    
    if sampling_ops_per_dim is None:
        eigs = matrix_size_dependent_sampling(
            matrix, isconv2d, conv_norm,
            num_row_samples=num_row_samples, 
            Q_ratio=Q_ratio, 
            step_size=step_size,
        )
    else:
        s = matrix.shape
        print(f"Matrix shape: {s}")
        eigs = fixed_sampling(
            matrix, isconv2d, conv_norm,
            num_row_samples=num_row_samples, 
            Q_ratio=Q_ratio, 
            sampling_ops_per_dim=sampling_ops_per_dim,
        )
    return eigs






######################################################################################################################
########################################## Unused Experimental Part ##################################################
######################################################################################################################

def matrix_size_dependent_sampling_conv2D(matrix, isconv2d, conv_norm, num_row_samples, Q_ratio, step_size=10):
    # Assuming matrix is of shape (out_channels, in_channels, kernel_height, kernel_width)
    out_channels, in_channels, kernel_height, kernel_width = matrix.shape
    num_col_samples = int(num_row_samples * Q_ratio)

    # Calculate number of sliding windows
    num_row_windows = max(1, (kernel_height - num_row_samples) // step_size + 1)
    num_col_windows = max(1, (kernel_width - num_col_samples) // step_size + 1)

    all_eigs = []

    for i in range(num_row_windows):
        for j in range(num_col_windows):
            # Extract submatrix
            row_start = i * step_size
            col_start = j * step_size
            submatrix = matrix[:, :, row_start:row_start + num_row_samples, col_start:col_start + num_col_samples]

             # Flatten to (out_channels, in_channels * kernel_height * kernel_width)
            if isconv2d:
                submatrix = submatrix.flatten(start_dim=1) * math.sqrt(conv_norm)
            eigs = torch.square(torch.linalg.svdvals(submatrix))
            all_eigs.append(eigs)

    all_eigs = torch.cat(all_eigs)
    all_eigs, _ = torch.sort(all_eigs, descending=False)

    return all_eigs

def fixed_sampling_conv2D(matrix, isconv2d, conv_norm, num_row_samples, Q_ratio, sampling_ops_per_dim):
    # Assuming matrix is of shape (out_channels, in_channels, kernel_height, kernel_width)
    out_channels, in_channels, kernel_height, kernel_width = matrix.shape
    num_col_samples = int(num_row_samples * Q_ratio)

    # Calculate step sizes
    row_step = max(1, (kernel_height - num_row_samples) // (sampling_ops_per_dim - 1))
    col_step = max(1, (kernel_width - num_col_samples) // (sampling_ops_per_dim - 1))

    # Calculate number of windows
    num_row_windows = min(sampling_ops_per_dim, max(1, (kernel_height - num_row_samples) // row_step + 1))
    num_col_windows = min(sampling_ops_per_dim, max(1, (kernel_width - num_col_samples) // col_step + 1))

    all_eigs = []

    for i in range(num_row_windows):
        for j in range(num_col_windows):
            # Extract submatrix
            row_start = min(i * row_step, kernel_height - num_row_samples)
            col_start = min(j * col_step, kernel_width - num_col_samples)
            submatrix = matrix[:, :, row_start:row_start + num_row_samples, col_start:col_start + num_col_samples]

            # Flatten to (out_channels, in_channels * kernel_height * kernel_width)
            if isconv2d:
                submatrix = submatrix.flatten(start_dim=1) * math.sqrt(conv_norm)
            eigs = torch.square(torch.linalg.svdvals(submatrix))
            all_eigs.append(eigs)

    all_eigs = torch.cat(all_eigs)
    all_eigs, _ = torch.sort(all_eigs, descending=False)

    return all_eigs


######################### FIXED SAMPLING METHOD FOR CONV2D LAYER SLICES ################################
def sample_conv2d(matrix, conv_norm, Q_ratio, num_row_samples, sampling_ops_per_dim):
    Wmats, N, M, rf = conv2D_Wmats(matrix, channels=CHANNELS.UNKNOWN)
    all_eigs = []
    for W in Wmats:
        rows, cols = W.shape
        
        ###notice that rows > cols  
        if rows == cols:
            if Q_ratio > 1:
                num_row_samples = rows // Q_ratio
                num_col_samples = cols
            else:
                num_row_samples = rows
                num_col_samples = cols // (1 / Q_ratio)

            # Ensure we don't sample more rows or columns than exist
            num_row_samples = max(1, int(num_row_samples))
            num_col_samples = max(1, int(num_col_samples))

            submatrix = W[:num_row_samples, :num_col_samples]
            submatrix *= math.sqrt(conv_norm)

            eigs = torch.square(torch.linalg.svdvals(submatrix))
            all_eigs.append(eigs)
        else:
            ####make it into a square matrix
            if Q_ratio > 1:
                num_row_samples = cols // Q_ratio
                num_col_samples = cols
            else:
                num_row_samples = cols
                num_col_samples = cols // (1 / Q_ratio)
                
            num_row_samples = max(1, int(num_row_samples))
            num_col_samples = max(1, int(num_col_samples))
            division = int(rows // cols)
            for i in range(division):
                submatrix = W[i*num_row_samples:(i+1)*num_row_samples, :num_col_samples]
                submatrix *= math.sqrt(conv_norm)
                eigs = torch.square(torch.linalg.svdvals(submatrix))
                all_eigs.append(eigs)
           
            
            
    all_eigs = torch.cat(all_eigs)
    all_eigs = torch.sort(all_eigs, descending=False).values

    return all_eigs
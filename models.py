import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math
from scipy import interpolate


def psnr(target, ref):
    """Calculate PSNR between target and reference images"""
    target_data = np.array(target, dtype=float)
    ref_data = np.array(ref, dtype=float)
    
    diff = ref_data - target_data
    diff = diff.flatten('C')
    
    rmse = math.sqrt(np.mean(diff ** 2.))
    
    return 20 * math.log10(255. / rmse)


def interpolation(noisy, SNR, Number_of_pilot, interp):
    """Interpolate noisy channel estimates"""
    noisy_image = np.zeros((40000, 72, 14, 2))
    
    noisy_image[:, :, :, 0] = np.real(noisy)
    noisy_image[:, :, :, 1] = np.imag(noisy)
    
    # Define pilot positions based on number of pilots
    if Number_of_pilot == 48:
        idx = [14*i for i in range(1, 72, 6)] + [4+14*(i) for i in range(4, 72, 6)] + \
              [7+14*(i) for i in range(1, 72, 6)] + [11+14*(i) for i in range(4, 72, 6)]
    elif Number_of_pilot == 16:
        idx = [4+14*(i) for i in range(1, 72, 9)] + [9+14*(i) for i in range(4, 72, 9)]
    elif Number_of_pilot == 24:
        idx = [14*i for i in range(1, 72, 9)] + [6+14*i for i in range(4, 72, 9)] + \
              [11+14*i for i in range(1, 72, 9)]
    elif Number_of_pilot == 8:
        idx = [4+14*(i) for i in range(5, 72, 18)] + [9+14*(i) for i in range(8, 72, 18)]
    elif Number_of_pilot == 36:
        idx = [14*(i) for i in range(1, 72, 6)] + [6+14*(i) for i in range(4, 72, 6)] + \
              [11+14*i for i in range(1, 72, 6)]
    
    r = [x // 14 for x in idx]
    c = [x % 14 for x in idx]
    
    interp_noisy = np.zeros((40000, 72, 14, 2))
    
    for i in range(len(noisy)):
        # Interpolate real part
        z = [noisy_image[i, j, k, 0] for j, k in zip(r, c)]
        if interp == 'rbf':
            f = interpolate.Rbf(np.array(r).astype(float), np.array(c).astype(float), 
                               z, function='gaussian')
            X, Y = np.meshgrid(range(72), range(14))
            z_intp = f(X, Y)
            interp_noisy[i, :, :, 0] = z_intp.T
        elif interp == 'spline':
            tck = interpolate.bisplrep(np.array(r).astype(float), np.array(c).astype(float), z)
            z_intp = interpolate.bisplev(range(72), range(14), tck)
            interp_noisy[i, :, :, 0] = z_intp
        
        # Interpolate imaginary part
        z = [noisy_image[i, j, k, 1] for j, k in zip(r, c)]
        if interp == 'rbf':
            f = interpolate.Rbf(np.array(r).astype(float), np.array(c).astype(float), 
                               z, function='gaussian')
            X, Y = np.meshgrid(range(72), range(14))
            z_intp = f(X, Y)
            interp_noisy[i, :, :, 1] = z_intp.T
        elif interp == 'spline':
            tck = interpolate.bisplrep(np.array(r).astype(float), np.array(c).astype(float), z)
            z_intp = interpolate.bisplev(range(72), range(14), tck)
            interp_noisy[i, :, :, 1] = z_intp
    
    interp_noisy = np.concatenate((interp_noisy[:, :, :, 0], interp_noisy[:, :, :, 1]), 
                                  axis=0).reshape(80000, 72, 14, 1)
    
    return interp_noisy


class SRCNN(nn.Module):
    """Super-Resolution CNN for channel estimation"""
    def __init__(self):
        super(SRCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using He normal initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class DNCNN(nn.Module):
    """Denoising CNN for channel estimation"""
    def __init__(self, depth=20, n_channels=64):
        super(DNCNN, self).__init__()
        
        layers = []
        
        # First layer: Conv + ReLU
        layers.append(nn.Conv2d(1, n_channels, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        
        # Middle layers: Conv + BN + ReLU
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=1e-3))
            layers.append(nn.ReLU(inplace=True))
        
        # Last layer: Conv
        layers.append(nn.Conv2d(n_channels, 1, kernel_size=3, padding=1, bias=True))
        
        self.dncnn = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        noise = self.dncnn(x)
        return x - noise  # Residual learning: input - noise


def SRCNN_train(train_data, train_label, val_data, val_label, 
                channel_model, num_pilots, SNR, epochs=300, batch_size=128, 
                device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Train SRCNN model"""
    
    # Convert numpy arrays to PyTorch tensors
    train_data = torch.FloatTensor(train_data).permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
    train_label = torch.FloatTensor(train_label).permute(0, 3, 1, 2)
    val_data = torch.FloatTensor(val_data).permute(0, 3, 1, 2)
    val_label = torch.FloatTensor(val_label).permute(0, 3, 1, 2)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(train_data, train_label)
    val_dataset = TensorDataset(val_data, val_label)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = SRCNN().to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
    
    best_val_loss = float('inf')
    
    print(f"Training SRCNN on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * data.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item() * data.size(0)
        
        val_loss /= len(val_loader.dataset)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"SRCNN_{channel_model}_{num_pilots}_{SNR}.pth")
            print(f'Saved best model with val_loss: {val_loss:.6f}')
    
    print(f"Training completed. Best validation loss: {best_val_loss:.6f}")


def SRCNN_predict(input_data, channel_model, num_pilots, SNR, 
                  batch_size=128,
                  device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Predict using trained SRCNN model with batch processing"""
    
    # Convert numpy array to PyTorch tensor
    if isinstance(input_data, np.ndarray):
        input_tensor = torch.FloatTensor(input_data).permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
    else:
        input_tensor = input_data
    
    # Create dataset and dataloader for batch processing
    dataset = TensorDataset(input_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Load model
    model = SRCNN().to(device)
    model.load_state_dict(torch.load(f"SRCNN_{channel_model}_{num_pilots}_{SNR}.pth"))
    model.eval()
    
    # Predict in batches
    predictions = []
    with torch.no_grad():
        for (batch_data,) in dataloader:
            batch_data = batch_data.to(device)
            output = model(batch_data)
            predictions.append(output.cpu())
    
    # Concatenate all predictions
    all_predictions = torch.cat(predictions, dim=0)
    
    # Convert back to numpy (N, C, H, W) -> (N, H, W, C)
    predicted = all_predictions.permute(0, 2, 3, 1).numpy()
    
    return predicted


def DNCNN_train(train_data, train_label, val_data, val_label, 
                channel_model, num_pilots, SNR, epochs=200, batch_size=128,
                device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Train DNCNN model"""
    
    # Convert numpy arrays to PyTorch tensors
    train_data = torch.FloatTensor(train_data).permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
    train_label = torch.FloatTensor(train_label).permute(0, 3, 1, 2)
    val_data = torch.FloatTensor(val_data).permute(0, 3, 1, 2)
    val_label = torch.FloatTensor(val_label).permute(0, 3, 1, 2)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(train_data, train_label)
    val_dataset = TensorDataset(val_data, val_label)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = DNCNN(depth=20, n_channels=64).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
    
    best_val_loss = float('inf')
    
    print(f"Training DNCNN on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * data.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item() * data.size(0)
        
        val_loss /= len(val_loader.dataset)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"DNCNN_{channel_model}_{num_pilots}_{SNR}.pth")
            print(f'Saved best model with val_loss: {val_loss:.6f}')
    
    print(f"Training completed. Best validation loss: {best_val_loss:.6f}")


def DNCNN_predict(input_data, channel_model, num_pilots, SNR,
                  batch_size=128,
                  device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Predict using trained DNCNN model with batch processing"""
    
    # Convert numpy array to PyTorch tensor
    if isinstance(input_data, np.ndarray):
        input_tensor = torch.FloatTensor(input_data).permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
    else:
        input_tensor = input_data
    
    # Create dataset and dataloader for batch processing
    dataset = TensorDataset(input_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Load model
    model = DNCNN(depth=20, n_channels=64).to(device)
    model.load_state_dict(torch.load(f"DNCNN_{channel_model}_{num_pilots}_{SNR}.pth"))
    model.eval()
    
    # Predict in batches
    predictions = []
    with torch.no_grad():
        for (batch_data,) in dataloader:
            batch_data = batch_data.to(device)
            output = model(batch_data)
            predictions.append(output.cpu())
    
    # Concatenate all predictions
    all_predictions = torch.cat(predictions, dim=0)
    
    # Convert back to numpy (N, C, H, W) -> (N, H, W, C)
    predicted = all_predictions.permute(0, 2, 3, 1).numpy()
    
    return predicted
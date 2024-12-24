import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from skimage.color import rgb2gray

def gaussian_2d(x, y, xi, yi, sigma, alpha):
    return alpha * np.exp(-((x - xi) ** 2 + (y - yi) ** 2) / (2 * sigma ** 2))

def mse(e1, e2):
    return np.mean((e1 - e2) ** 2)

def add_gaussians(A, N, T):
    # A is the picture, N is number and T is squared error
    rows, cols = A.shape
    B = np.zeros_like(A)
    mse_values = []
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
    for i in range(N):
        def loss(params):
            xi, yi, sigma, alpha = params
            G = gaussian_2d(X, Y, xi, yi, sigma, alpha)
            return mse(A, B + G)
        initial_params = [np.random.uniform(0, cols),
                          np.random.uniform(0, rows),
                          np.random.uniform(5, 20),
                          np.random.uniform(0.1, 0.5)]

        result = minimize(loss, initial_params, bounds=[(0, cols),(0, rows),(3, 20),(0.01, 1)])
        if result.success:
            xi, yi, sigma, alpha = result.x
            G = gaussian_2d(X,Y,xi,yi,sigma,alpha)
            B += G
            current_mse = mse(A, B)
            mse_values.append(current_mse)
            print(f"Iteration {i + 1}: MSE = {current_mse:.6f}")
            if current_mse < T:
                print(f"Stopping early at iteration {i + 1}, MSE = {current_mse:.4f}")
                break
    return B, mse_values

def picture_deal(img,N,T):
    R_channel = img[:, :, 0] / 255.0
    G_channel = img[:, :, 1] / 255.0
    B_channel = img[:, :, 2] / 255.0
    if img.ndim == 3:
        grayscale = rgb2gray(img)
    else:
        grayscale = img / 255.0
    A = grayscale / grayscale.max()
    N_values = N
    T /= 255
    B, mse_values = add_gaussians(A, N_values, T)
    reconstructed_image = np.stack((B * (R_channel / A), B * (G_channel / A), B * (B_channel / A)), axis=-1)
    reconstructed_image = np.clip(reconstructed_image, 0, 1)
    return reconstructed_image,mse_values

def showmse(mse_values):
    plt.plot(range(1, len(mse_values) + 1), mse_values, marker='o')
    plt.title("MSE vs N")
    plt.xlabel("N")
    plt.ylabel("MSE")
    plt.grid()
    plt.show()

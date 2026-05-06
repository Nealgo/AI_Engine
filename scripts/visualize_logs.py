import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def plot_log(csv_file):
    df = pd.read_csv(csv_file)
    
    epochs = df['Epoch']
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Training Metrics for {os.path.basename(csv_file)}')
    
    # Train Loss
    axs[0, 0].plot(epochs, df['Train Loss'], label='Train Loss', color='tab:red')
    axs[0, 0].set_title('Train Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].grid(True)
    
    # Test PSNR
    axs[0, 1].plot(epochs, df['Test PSNR'], label='Test PSNR', color='tab:blue')
    axs[0, 1].set_title('Test PSNR')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('PSNR (dB)')
    axs[0, 1].grid(True)
    
    # Test SSIM
    axs[1, 0].plot(epochs, df['Test SSIM'], label='Test SSIM', color='tab:green')
    axs[1, 0].set_title('Test SSIM')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('SSIM')
    axs[1, 0].grid(True)
    
    # Learning Rate
    axs[1, 1].plot(epochs, df['Learning Rate'], label='Learning Rate', color='tab:orange')
    axs[1, 1].set_title('Learning Rate')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('LR')
    axs[1, 1].set_yscale('log')
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    
    # Save the plot
    save_path = csv_file.replace('.csv', '.png')
    plt.savefig(save_path, dpi=300)
    print(f"Saved visualization to {save_path}")
    
    # If in interactive mode, display it, else comment out if running on headless server:
    # plt.show()
    plt.close()

if __name__ == '__main__':
    log_files = glob.glob('log/*.csv')
    if not log_files:
        print("No log files found in the 'log' directory.")
    else:
        for file in log_files:
            plot_log(file)

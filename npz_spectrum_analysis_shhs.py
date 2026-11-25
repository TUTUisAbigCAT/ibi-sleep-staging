import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq
import seaborn as sns
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 确保 MultipleLocator 已导入
from matplotlib.ticker import MultipleLocator

class IBISpectrumAnalyzer:
    def __init__(self, npz_file_path):
        self.npz_file_path = npz_file_path
        self.data = None
        self.fs = None
        self.ibi = None
        self.stages = None
        self.ahi = None
        self.group_spectra = {}
        self.group_frequencies = {}
        
    def load_data(self):
        """加载npz文件数据"""
        try:
            data = np.load(self.npz_file_path)
            self.ibi = data['data']
            self.fs = int(data['fs'])
            self.stages = data['stages']
            self.ahi = float(data['ahi'])

            num_samples = int((self.fs / 25) * 30)
            
            if num_samples < 30: 
                 num_samples = max(int(self.fs * 30), 30)

            num_segments = len(self.ibi) // num_samples

            ibi = self.ibi
            ibi = ibi[:num_samples*num_segments]
            ibi = ibi.reshape(num_segments, num_samples)

            self.ibi = ibi

            labels = self.stages
            labels = labels[::num_samples]
            labels = labels[:num_segments]

            self.stages = labels
            
        except Exception as e:
            return False
        return True
    
    def group_data_by_stages(self):
        """根据stages分组IBI数据"""
        if self.ibi is None or self.stages is None:
            return None
            
        min_length = min(self.ibi.shape[0], len(self.stages))
        ibi_trimmed = self.ibi[:min_length]
        stages_trimmed = self.stages[:min_length]
        
        grouped_data = defaultdict(list)
        
        for segment, stage in zip(ibi_trimmed, stages_trimmed):
            grouped_data[stage].append(segment)
            
        self.grouped_data = dict(grouped_data)
        
        return self.grouped_data
    
    def calculate_spectrum(self, signal_segment, method='fft'):
        """
        计算单个信号段的频谱
        """
        if len(signal_segment) < 10:
            return None, None, None
            
        try:
            n = len(signal_segment)
            fft_vals = fft(signal_segment)
            freqs = fftfreq(n, 1/self.fs)
            freqs = freqs[:n//2]
            fft_vals = fft_vals[:n//2]
            
            magnitude = np.abs(fft_vals)
            phase = np.angle(fft_vals)
            
        except Exception as e:
            return None, None, None
            
        return freqs, magnitude, phase

def batch_average_spectra(directory_path, method='fft', file_pattern='*.npz', 
                          save_path='batch_averaged_spectrum.png'):
    """
    批量处理目录中所有文件的所有segment并绘制平均频谱
    """
    directory_path = Path(directory_path)
    npz_files = list(directory_path.glob(file_pattern))
    
    if not npz_files:
        print(f"在 {directory_path} 中未找到匹配 {file_pattern} 的文件")
        return None
    
    print(f"找到 {len(npz_files)} 个文件待处理\n")
    
    all_segment_magnitudes = defaultdict(list)
    all_segment_phases = defaultdict(list)
    all_segment_frequencies = defaultdict(list)
    processed_count = 0
    
    for filepath in tqdm(npz_files, desc="处理文件"):
        try:
            analyzer = IBISpectrumAnalyzer(filepath)
            
            if not analyzer.load_data():
                continue
            
            analyzer.group_data_by_stages()
            
            for stage, segments in analyzer.grouped_data.items():
                for segment in segments:
                    freqs, magnitude, phase = analyzer.calculate_spectrum(segment, method)
                    if freqs is not None and magnitude is not None:
                        all_segment_magnitudes[stage].append(magnitude)
                        all_segment_phases[stage].append(phase)
                        all_segment_frequencies[stage].append(freqs)
            
            processed_count += 1
                
        except Exception as e:
            print(f"处理文件 {filepath.name} 时出错: {e}")
            continue
    
    print(f"\n成功处理 {processed_count} 个文件")
    
    if processed_count == 0:
        print("没有成功处理的文件，无法生成平均频谱")
        return None
    
    averaged_spectra = {}
    averaged_frequencies = {}
    
    print("\n计算所有segment的平均频谱...")
    for stage in all_segment_magnitudes.keys():
        magnitudes = all_segment_magnitudes[stage]
        phases = all_segment_phases[stage]
        freqs_list = all_segment_frequencies[stage]
        
        if not magnitudes:
            continue
        
        common_freqs = freqs_list[0]
        interpolated_magnitudes = []
        interpolated_phases = []
        for magnitude, phase, freqs in zip(magnitudes, phases, freqs_list):
            if len(freqs) != len(common_freqs):
                magnitude_interp = np.interp(common_freqs, freqs, magnitude)
                phase_interp = np.interp(common_freqs, freqs, phase)
                interpolated_magnitudes.append(magnitude_interp)
                interpolated_phases.append(phase_interp)
            else:
                interpolated_magnitudes.append(magnitude)
                interpolated_phases.append(phase)
        
        min_length = min(len(mag) for mag in interpolated_magnitudes)
        interpolated_magnitudes = [mag[:min_length] for mag in interpolated_magnitudes]
        interpolated_phases = [ph[:min_length] for ph in interpolated_phases]
        common_freqs = common_freqs[:min_length]
        
        interpolated_magnitudes = np.array(interpolated_magnitudes)
        interpolated_phases = np.array(interpolated_phases)
        mean_magnitude = np.mean(interpolated_magnitudes, axis=0)
        std_magnitude = np.std(interpolated_magnitudes, axis=0)
        mean_phase = np.mean(interpolated_phases, axis=0)
        std_phase = np.std(interpolated_phases, axis=0)
        
        averaged_frequencies[stage] = common_freqs
        averaged_spectra[stage] = {
            'mean_magnitude': mean_magnitude,
            'std_magnitude': std_magnitude,
            'mean_phase': mean_phase,
            'std_phase': std_phase,
            'n_segments': len(interpolated_magnitudes)
        }
        
    
    # 绘制平均频谱 - 每个stage两个子图（幅度和相位）
    n_groups = len(averaged_spectra)
    
    # 1. 图表再大一些: figsize=(8*n_groups, 12)
    fig, axes = plt.subplots(2, n_groups, figsize=(8*n_groups, 12)) 
    
    if n_groups == 1:
        axes = axes.reshape(2, 1)
    
    colors = plt.cm.Set1(np.linspace(0, 1, n_groups))
    
    # 2. X 轴刻度: 1.0 Hz 间隔
    major_x_locator = MultipleLocator(1.0)
    # 3. Y 轴刻度: 幅度 Y 轴设置为 2.0 间隔
    major_y_magnitude_locator = MultipleLocator(2.0)
    # 4. 相位 Y 轴刻度: 设置为 0.1 间隔 (更精细)
    major_y_phase_locator = MultipleLocator(0.1)
    
    for i, (stage, spectra_data) in enumerate(averaged_spectra.items()):
        freqs = averaged_frequencies[stage]
        mean_magnitude = spectra_data['mean_magnitude']
        std_magnitude = spectra_data['std_magnitude']
        mean_phase = spectra_data['mean_phase']
        n_segments = spectra_data['n_segments']
        
        # --- 幅度谱 (Magnitude Spectrum) ---
        ax_mag = axes[0, i]
        ax_mag.plot(freqs, mean_magnitude, color=colors[i], linewidth=2, label='Mean Magnitude')
        
        # 标记标准差范围 (平均值 +/- 1 * 标准差)
        ax_mag.fill_between(
            freqs, 
            mean_magnitude - std_magnitude, 
            mean_magnitude + std_magnitude, 
            color=colors[i], 
            alpha=0.2, 
            label='$\pm 1 \sigma$'
        )
        
        ax_mag.set_xlabel('Frequency (Hz)')
        ax_mag.set_ylabel('Magnitude')
        ax_mag.set_title(f'Stage {stage} - Magnitude Spectrum\n({processed_count} Files, {n_segments} Segments)')
        ax_mag.grid(True, alpha=0.3)
        ax_mag.legend() 
        
        ax_mag.xaxis.set_major_locator(major_x_locator)
        ax_mag.yaxis.set_major_locator(major_y_magnitude_locator) # <-- 幅度 Y 轴刻度间隔 = 2.0
        
        # --- 相位谱 (Phase Spectrum) ---
        ax_phase = axes[1, i]
        ax_phase.plot(freqs, mean_phase, color=colors[i], linewidth=2)
        ax_phase.set_xlabel('Frequency (Hz)')
        ax_phase.set_ylabel('Phase (rad)')
        ax_phase.set_title(f'Stage {stage} - Phase Spectrum')
        ax_phase.grid(True, alpha=0.3)
        ax_phase.xaxis.set_major_locator(major_x_locator)
        ax_phase.yaxis.set_major_locator(major_y_phase_locator) # <-- 相位 Y 轴刻度间隔 = 0.1 (更精细)
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n批量平均频谱图已保存到: {save_path}")
    
    return averaged_spectra, averaged_frequencies

if __name__ == "__main__":

    # 批量平均处理示例（新增功能）
    root_path = Path("/home/yxtao/scratch/WatchSleepProcessedDataset/SHHS_MESA_IBI")
    batch_average_spectra(
        directory_path=root_path,
        method='periodogram',
        file_pattern='shhs-*.npz',
        save_path='batch_averaged_spectrum_shhs.png'
    )
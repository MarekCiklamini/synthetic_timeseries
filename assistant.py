# -*- coding: utf-8 -*-
"""
"""
# %%
import numpy as np
import pandas as pd
import glob
import dash
from dash import dcc  # import dash_core_components as dcc
from dash import html # import dash_html_components as html ... instead 
import plotly.express as px
# from pptx.util import Mm
# from pptx import Presentation
# %% Class T
class T:
    # initiate serie 
    def __init__(self):
        self.tlength = 1100
        self.x = np.arange(self.tlength,dtype=int)
        self.Tsin = np.sin(0.015 * self.x) *1.25 
        self.df = pd.DataFrame({'T': self.x, 'y': self.Tsin})
        
        # Multivariate components (price, volume, volatility state)
        self.price = self.Tsin.copy()
        self.volume = None
        self.volatility_state = None
        
    def add_noise(self, noiseWidth):
        self.noise = np.random.rand(self.tlength,) * noiseWidth
        self.Tsin = self.Tsin + self.noise
        self.price = self.Tsin.copy()
        self._update_dataframe()
        
    def add_sinSerie(self, a, b,):
        Tsin1 = np.sin(a * self.x) * b
        self.Tsin = self.Tsin + Tsin1
        self.price = self.Tsin.copy()
        self._update_dataframe()
        
    def add_eiwSerie(self, w , d, ): 
        # https://stackoverflow.com/questions/24592803/separate-real-and-imaginary-part-of-a-complex-number-in-python
        Teiw = np.exp((1j * w + d ) * self.x)  
        self.Tsin = self.Tsin + Teiw.real
        self.price = self.Tsin.copy()
        self._update_dataframe()
    
    def add_volume_state_space(self, base_volume=1000, volatility_impact=0.5, 
                               trend_following=0.3, mean_reversion=0.1):
        """
        Generate volume using a simple state space model:
        Volume correlates with:
        1. Price volatility (higher vol = higher volume)
        2. Price momentum (trend following behavior)
        3. Mean reversion component
        
        State space representation:
        Volume[t] = base + α*|returns[t]| + β*momentum[t] + γ*reversion[t] + noise
        
        Parameters:
        -----------
        base_volume : float
            Base volume level
        volatility_impact : float
            How much volatility increases volume (0-1)
        trend_following : float
            Volume increases with strong trends (0-1)
        mean_reversion : float
            Volume spikes at extremes (0-1)
        """
        price = self.price
        
        # Calculate price features
        returns = np.diff(price, prepend=price[0])
        volatility = np.abs(returns)
        
        # Momentum (moving average of returns)
        momentum = np.convolve(returns, np.ones(10)/10, mode='same')
        
        # Distance from moving average (mean reversion signal)
        ma = np.convolve(price, np.ones(50)/50, mode='same')
        deviation = np.abs(price - ma) / (ma + 1e-8)
        
        # State space volume model
        volume = (base_volume + 
                 volatility_impact * volatility * base_volume +
                 trend_following * np.abs(momentum) * base_volume +
                 mean_reversion * deviation * base_volume)
        
        # Add noise
        volume += np.random.normal(0, base_volume * 0.1, self.tlength)
        volume = np.maximum(volume, base_volume * 0.1)  # Ensure positive
        
        self.volume = volume
        self.volatility_state = volatility
        self._update_dataframe()
        
        return volume
    
    def add_volume_garch_style(self, base_volume=1000, alpha=0.1, beta=0.85):
        """
        GARCH-style volume with volatility clustering:
        Volume variance follows autoregressive process
        
        Vol[t] = base + α*shock[t-1] + β*Vol[t-1] + noise
        """
        price = self.price
        returns = np.diff(price, prepend=price[0])
        
        volume = np.zeros(self.tlength)
        volume[0] = base_volume
        
        shock = np.abs(returns) * base_volume
        
        for t in range(1, self.tlength):
            volume[t] = (base_volume * 0.3 + 
                        alpha * shock[t-1] + 
                        beta * volume[t-1] +
                        np.random.normal(0, base_volume * 0.05))
            volume[t] = max(volume[t], base_volume * 0.1)
        
        self.volume = volume
        self._update_dataframe()
        
        return volume
    
    def add_volume_regime_switching(self, base_volume=1000, n_regimes=3):
        """
        Volume with regime switching (low/medium/high activity regimes)
        Uses hidden Markov-style state transitions
        
        Regimes:
        0 = Low activity (60% of base)
        1 = Normal activity (100% of base)
        2 = High activity (200% of base)
        """
        # Transition probability matrix (rows = from, cols = to)
        P = np.array([
            [0.95, 0.04, 0.01],  # Low tends to stay low
            [0.10, 0.80, 0.10],  # Normal balanced
            [0.05, 0.15, 0.80]   # High tends to stay high
        ])
        
        volume_multipliers = np.array([0.6, 1.0, 2.0])
        
        # Start in normal regime
        current_regime = 1
        regimes = np.zeros(self.tlength, dtype=int)
        volume = np.zeros(self.tlength)
        
        for t in range(self.tlength):
            regimes[t] = current_regime
            
            # Volume depends on regime + price action
            price_vol = np.abs(np.diff(self.price[max(0,t-5):t+1])).mean() if t > 0 else 0
            
            volume[t] = (base_volume * volume_multipliers[current_regime] * 
                        (1 + price_vol) +
                        np.random.normal(0, base_volume * 0.1))
            volume[t] = max(volume[t], base_volume * 0.1)
            
            # Transition to next regime
            current_regime = np.random.choice(n_regimes, p=P[current_regime])
        
        self.volume = volume
        self.regime = regimes
        self._update_dataframe()
        
        return volume
    
    def add_volume_microstructure(self, base_volume=1000, informed_ratio=0.2):
        """
        Market microstructure model with informed/uninformed traders
        
        Volume = Uninformed + Informed
        - Uninformed: constant flow + noise
        - Informed: spikes around price changes (information events)
        """
        price = self.price
        returns = np.diff(price, prepend=price[0])
        
        # Uninformed traders (constant + noise)
        uninformed = base_volume * (1 - informed_ratio) + np.random.normal(0, base_volume * 0.05, self.tlength)
        
        # Informed traders (spike on information events)
        # Detect large price moves as information events
        abs_returns = np.abs(returns)
        threshold = np.percentile(abs_returns, 80)
        information_events = abs_returns > threshold
        
        informed = np.zeros(self.tlength)
        for t in range(self.tlength):
            if information_events[t]:
                # Spike over several periods
                for lag in range(5):
                    if t + lag < self.tlength:
                        informed[t + lag] += base_volume * informed_ratio * (1 - lag * 0.15)
        
        informed += np.random.exponential(base_volume * informed_ratio * 0.1, self.tlength)
        
        volume = uninformed + informed
        self.volume = volume
        self._update_dataframe()
        
        return volume
    
    def add_bid_ask_spread(self, base_spread=0.01, distribution='uniform', 
                           volatility_scaling=True, **kwargs):
        """
        Generate bid/ask prices with configurable spread distributions.
        Assumes price column is the mid price.
        
        Parameters:
        -----------
        base_spread : float
            Base spread as percentage of price (0.01 = 1%)
        distribution : str
            'uniform', 'normal', 'exponential', 'gamma', 'lognormal', 'weibull'
        volatility_scaling : bool
            Scale spread with price volatility (wider spreads in volatile periods)
        **kwargs : dict
            Distribution-specific parameters
        
        Returns:
        --------
        tuple: (bid, ask, spread)
        """
        price = self.price
        
        # Calculate base spread
        if volatility_scaling:
            returns = np.diff(price, prepend=price[0])
            volatility = np.abs(returns)
            # Rolling volatility estimate
            vol_ma = np.convolve(volatility, np.ones(20)/20, mode='same')
            vol_scaling = 1 + vol_ma / (vol_ma.mean() + 1e-8)
        else:
            vol_scaling = np.ones(self.tlength)
        
        # Generate spread using specified distribution
        if distribution == 'uniform':
            # Uniform spread around base
            width = kwargs.get('width', 0.5)  # +/- 50% of base
            spread_multiplier = np.random.uniform(
                1 - width, 1 + width, self.tlength
            )
        
        elif distribution == 'normal':
            # Normal distribution (can go negative, clipped)
            std = kwargs.get('std', 0.3)
            spread_multiplier = np.random.normal(1.0, std, self.tlength)
            spread_multiplier = np.maximum(spread_multiplier, 0.1)
        
        elif distribution == 'exponential':
            # Exponential (right-skewed, occasional wide spreads)
            scale = kwargs.get('scale', 0.5)
            spread_multiplier = np.random.exponential(scale, self.tlength) + 0.5
        
        elif distribution == 'gamma':
            # Gamma distribution (flexible shape)
            shape = kwargs.get('shape', 2.0)
            scale = kwargs.get('scale', 0.5)
            spread_multiplier = np.random.gamma(shape, scale, self.tlength)
        
        elif distribution == 'lognormal':
            # Log-normal (realistic for financial spreads)
            mean = kwargs.get('mean', 0.0)
            sigma = kwargs.get('sigma', 0.3)
            spread_multiplier = np.random.lognormal(mean, sigma, self.tlength)
        
        elif distribution == 'weibull':
            # Weibull (can model fat tails)
            shape = kwargs.get('shape', 1.5)
            spread_multiplier = np.random.weibull(shape, self.tlength)
        
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        # Calculate half-spread
        half_spread = price * base_spread * spread_multiplier * vol_scaling / 2
        
        # Generate bid/ask
        bid = price - half_spread
        ask = price + half_spread
        spread = ask - bid
        
        # Store
        self.bid = bid
        self.ask = ask
        self.spread = spread
        self.half_spread = half_spread
        
        self._update_dataframe()
        
        return bid, ask, spread
    
    def add_bid_ask_order_flow(self, base_spread=0.01, imbalance_impact=0.5):
        """
        Generate bid/ask with order flow imbalance effects.
        Buy pressure → ask closer to mid, bid wider
        Sell pressure → bid closer to mid, ask wider
        
        Parameters:
        -----------
        base_spread : float
            Minimum spread as percentage
        imbalance_impact : float
            How much order flow affects spread asymmetry (0-1)
        """
        price = self.price
        
        if self.volume is None:
            raise ValueError("Volume must be generated first (use add_volume_* methods)")
        
        volume = self.volume
        returns = np.diff(price, prepend=price[0])
        
        # Estimate order flow imbalance from price-volume relationship
        # Positive imbalance = buying pressure, negative = selling pressure
        signed_volume = np.sign(returns) * volume
        imbalance = np.convolve(signed_volume, np.ones(10)/10, mode='same')
        
        # Normalize imbalance
        imbalance = imbalance / (volume.mean() + 1e-8)
        imbalance = np.clip(imbalance, -1, 1)
        
        # Base half-spread
        base_half = price * base_spread / 2
        
        # Asymmetric adjustment based on imbalance
        # Buy pressure: tighter ask, wider bid
        # Sell pressure: tighter bid, wider ask
        bid_adjustment = base_half * (1 + imbalance * imbalance_impact)
        ask_adjustment = base_half * (1 - imbalance * imbalance_impact)
        
        bid = price - bid_adjustment
        ask = price + ask_adjustment
        spread = ask - bid
        
        self.bid = bid
        self.ask = ask
        self.spread = spread
        self.order_flow_imbalance = imbalance
        
        self._update_dataframe()
        
        return bid, ask, spread
    
    def add_bid_ask_inventory(self, base_spread=0.01, inventory_impact=0.3):
        """
        Market maker inventory-based spread model.
        High inventory → widen bid (discourage buying)
        Low inventory → widen ask (discourage selling)
        
        Parameters:
        -----------
        base_spread : float
            Base spread percentage
        inventory_impact : float
            Impact of inventory deviation (0-1)
        """
        price = self.price
        
        if self.volume is None:
            raise ValueError("Volume must be generated first")
        
        # Simulate market maker inventory (random walk bounded)
        inventory = np.zeros(self.tlength)
        inventory[0] = 0  # Start neutral
        
        returns = np.diff(price, prepend=price[0])
        
        for t in range(1, self.tlength):
            # Inventory changes with market direction (simplified)
            inventory_change = -np.sign(returns[t]) * np.random.exponential(0.5)
            inventory[t] = inventory[t-1] + inventory_change
            
            # Mean reversion (market maker trades to reduce inventory)
            inventory[t] -= inventory[t] * 0.05
        
        # Normalize inventory to [-1, 1]
        if inventory.std() > 0:
            inventory_norm = inventory / (2 * inventory.std())
            inventory_norm = np.clip(inventory_norm, -1, 1)
        else:
            inventory_norm = inventory
        
        # Spread adjustment
        base_half = price * base_spread / 2
        
        # Positive inventory → widen bid, tighten ask (want to sell)
        # Negative inventory → tighten bid, widen ask (want to buy)
        bid_adjustment = base_half * (1 + inventory_norm * inventory_impact)
        ask_adjustment = base_half * (1 - inventory_norm * inventory_impact)
        
        bid = price - bid_adjustment
        ask = price + ask_adjustment
        spread = ask - bid
        
        self.bid = bid
        self.ask = ask
        self.spread = spread
        self.inventory = inventory
        
        self._update_dataframe()
        
        return bid, ask, spread
    
    def _update_dataframe(self):
        """Update dataframe with all available series"""
        data = {'T': self.x, 'price': self.price}
        
        if self.volume is not None:
            data['volume'] = self.volume
        
        if self.volatility_state is not None:
            data['volatility'] = self.volatility_state
        
        if hasattr(self, 'regime'):
            data['regime'] = self.regime
        
        if hasattr(self, 'bid'):
            data['bid'] = self.bid
            data['ask'] = self.ask
            data['spread'] = self.spread
        
        if hasattr(self, 'order_flow_imbalance'):
            data['order_flow_imbalance'] = self.order_flow_imbalance
        
        if hasattr(self, 'inventory'):
            data['inventory'] = self.inventory
        
        # Keep backward compatibility
        data['y'] = self.Tsin
        
        self.df = pd.DataFrame(data)
    
    def plot_overview(self, figsize=(14, 10)):
        """
        Quick visualization of all generated series using pandas plot
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        """
        import matplotlib.pyplot as plt
        
        df = self.df
        
        # Determine number of subplots based on available data
        n_plots = 1  # Always have price
        if self.volume is not None:
            n_plots += 1
        if hasattr(self, 'bid'):
            n_plots += 1
        if hasattr(self, 'volatility_state'):
            n_plots += 1
        if hasattr(self, 'regime'):
            n_plots += 1
        
        fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
        
        # Make axes iterable if only one subplot
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # 1. Price with bid/ask if available
        ax = axes[plot_idx]
        df[['price']].plot(ax=ax, color='blue', linewidth=1.5, label='Mid Price')
        
        if hasattr(self, 'bid'):
            df[['bid']].plot(ax=ax, color='green', linewidth=1, alpha=0.7, label='Bid')
            df[['ask']].plot(ax=ax, color='red', linewidth=1, alpha=0.7, label='Ask')
            ax.fill_between(df.index, df['bid'], df['ask'], alpha=0.2, color='gray', label='Spread')
        
        ax.set_ylabel('Price', fontsize=10)
        ax.set_title('Price Series (with Bid/Ask if available)', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plot_idx += 1
        
        # 2. Volume
        if self.volume is not None:
            ax = axes[plot_idx]
            df[['volume']].plot(ax=ax, color='purple', linewidth=1, kind='area', alpha=0.5)
            ax.set_ylabel('Volume', fontsize=10)
            ax.set_title('Volume', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        
        # 3. Spread details (if bid/ask exist)
        if hasattr(self, 'bid'):
            ax = axes[plot_idx]
            df[['spread']].plot(ax=ax, color='orange', linewidth=1.5)
            ax.set_ylabel('Spread', fontsize=10)
            ax.set_title('Bid-Ask Spread', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add order flow imbalance if available
            if hasattr(self, 'order_flow_imbalance'):
                ax2 = ax.twinx()
                df[['order_flow_imbalance']].plot(ax=ax2, color='cyan', linewidth=1, 
                                                   alpha=0.6, style='--', label='Order Flow')
                ax2.set_ylabel('Order Flow Imbalance', fontsize=9, color='cyan')
                ax2.tick_params(axis='y', labelcolor='cyan')
                ax2.legend(loc='upper right')
            
            plot_idx += 1
        
        # 4. Volatility
        if hasattr(self, 'volatility_state'):
            ax = axes[plot_idx]
            df[['volatility']].plot(ax=ax, color='red', linewidth=1, kind='area', alpha=0.4)
            ax.set_ylabel('Volatility', fontsize=10)
            ax.set_title('Price Volatility (Returns)', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        
        # 5. Regime (if available)
        if hasattr(self, 'regime'):
            ax = axes[plot_idx]
            df[['regime']].plot(ax=ax, color='brown', linewidth=2, drawstyle='steps-post')
            ax.set_ylabel('Regime', fontsize=10)
            ax.set_title('Volume Regime', fontsize=12, fontweight='bold')
            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels(['Low', 'Normal', 'High'])
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        
        axes[-1].set_xlabel('Time', fontsize=10)
        
        plt.tight_layout()
        plt.show()
        
        return fig, axes
    
    def plot_market_depth(self, window=100, figsize=(12, 6)):
        """
        Visualize bid/ask spread evolution over a time window
        
        Parameters:
        -----------
        window : int
            Number of time periods to display
        figsize : tuple
            Figure size
        """
        import matplotlib.pyplot as plt
        
        if not hasattr(self, 'bid'):
            raise ValueError("Bid/ask data not available. Use add_bid_ask_* methods first.")
        
        df = self.df.iloc[-window:].copy()
        df = df.reset_index(drop=True)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Top: Price levels
        ax1.plot(df.index, df['price'], color='blue', linewidth=2, label='Mid Price')
        ax1.fill_between(df.index, df['bid'], df['ask'], alpha=0.3, color='gray', label='Bid-Ask Spread')
        ax1.plot(df.index, df['bid'], color='green', linewidth=1, alpha=0.7, label='Bid')
        ax1.plot(df.index, df['ask'], color='red', linewidth=1, alpha=0.7, label='Ask')
        ax1.set_ylabel('Price', fontsize=10)
        ax1.set_title('Market Depth View', fontsize=12, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Bottom: Spread width
        ax2.bar(df.index, df['spread'], color='orange', alpha=0.6, label='Spread')
        ax2.set_ylabel('Spread', fontsize=10)
        ax2.set_xlabel('Time', fontsize=10)
        ax2.set_title('Spread Width', fontsize=12, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
        
        return fig, (ax1, ax2)
    
    def plot_correlation_matrix(self, figsize=(10, 8)):
        """
        Plot correlation matrix of all available series
        """
        import matplotlib.pyplot as plt
        
        # Select numeric columns only
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['T', 'y']]
        
        if len(numeric_cols) < 2:
            print("Not enough variables for correlation matrix")
            return
        
        corr = self.df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap using matplotlib
        im = ax.imshow(corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(numeric_cols)))
        ax.set_yticks(np.arange(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=45, ha='right')
        ax.set_yticklabels(numeric_cols)
        
        # Add correlation values as text
        for i in range(len(numeric_cols)):
            for j in range(len(numeric_cols)):
                text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=9)
        
        ax.set_title('Correlation Matrix of Market Variables', fontsize=12, fontweight='bold')
        fig.colorbar(im, ax=ax, shrink=0.8)
        
        plt.tight_layout()
        plt.show()
        
        return fig, ax
    
    # ==================== PARAMETER EXTRACTION FROM REAL DATA ====================
    
    @staticmethod
    def fit_from_timeseries(filepath=None, df=None, price_col='price', 
                           volume_col='volume', time_col='time',
                           bid_col=None, ask_col=None):
        """
        Extract parameters from real time series data to replicate its characteristics.
        
        This is the main entry point for fitting synthetic data generator to real data.
        
        Parameters:
        -----------
        filepath : str
            Path to CSV file with real market data
        df : pd.DataFrame
            Or provide dataframe directly
        price_col : str
            Column name for price (or mid price)
        volume_col : str
            Column name for volume
        time_col : str
            Column name for timestamp
        bid_col, ask_col : str
            Column names for bid/ask if available
            
        Returns:
        --------
        dict : Fitted parameters and T instance configured to match real data
        """
        from scipy import signal, stats
        from scipy.optimize import curve_fit
        
        # Load data
        if df is None:
            df = pd.read_csv(filepath)
        
        # Extract price series
        if price_col not in df.columns and bid_col and ask_col:
            # Calculate mid price from bid/ask
            price = (df[bid_col] + df[ask_col]) / 2
        else:
            price = df[price_col].values
        
        results = {
            'length': len(price),
            'price_stats': {},
            'noise_params': {},
            'trend_params': {},
            'volume_params': {},
            'spread_params': {},
            'quality_report': {}
        }
        
        # ===== 1. Basic Statistics =====
        results['price_stats'] = {
            'mean': float(np.mean(price)),
            'std': float(np.std(price)),
            'min': float(np.min(price)),
            'max': float(np.max(price)),
            'median': float(np.median(price))
        }
        
        # ===== 2. Detrend and Extract Noise =====
        # Remove polynomial trend
        t_idx = np.arange(len(price))
        
        # Try different polynomial degrees
        best_poly_degree = None
        best_residual_std = np.inf
        
        for degree in [1, 2, 3, 4, 5]:
            coeffs = np.polyfit(t_idx, price, degree)
            poly_trend = np.polyval(coeffs, t_idx)
            residuals = price - poly_trend
            residual_std = np.std(residuals)
            
            if residual_std < best_residual_std:
                best_residual_std = residual_std
                best_poly_degree = degree
                best_coeffs = coeffs
                best_trend = poly_trend
        
        results['trend_params']['polynomial'] = {
            'degree': best_poly_degree,
            'coefficients': best_coeffs.tolist(),
            'r_squared': float(1 - np.var(price - best_trend) / np.var(price))
        }
        
        detrended = price - best_trend
        
        # ===== 3. Spectral Analysis (Fourier) =====
        # Find dominant frequencies using FFT
        fft = np.fft.fft(detrended)
        freqs = np.fft.fftfreq(len(detrended))
        power = np.abs(fft) ** 2
        
        # Get top 5 frequencies (excluding DC component)
        positive_freqs = freqs[freqs > 0]
        positive_power = power[freqs > 0]
        top_indices = np.argsort(positive_power)[-5:][::-1]
        
        sinusoidal_components = []
        for idx in top_indices:
            freq = positive_freqs[idx]
            amplitude = 2 * np.abs(fft[freqs > 0][idx]) / len(detrended)
            phase = np.angle(fft[freqs > 0][idx])
            sinusoidal_components.append({
                'frequency': float(freq),
                'amplitude': float(amplitude),
                'phase': float(phase),
                'power': float(positive_power[idx])
            })
        
        results['trend_params']['sinusoidal'] = sinusoidal_components
        
        # ===== 4. Noise Characterization =====
        # Remove sinusoidal components
        signal_reconstruction = np.zeros(len(price))
        for comp in sinusoidal_components[:3]:  # Use top 3
            signal_reconstruction += comp['amplitude'] * np.sin(
                2 * np.pi * comp['frequency'] * t_idx + comp['phase']
            )
        
        pure_noise = detrended - signal_reconstruction
        
        # Test for distribution
        noise_dist = T._fit_distribution(pure_noise)
        results['noise_params'] = noise_dist
        
        # ===== 5. Volume Analysis =====
        if volume_col in df.columns:
            volume = df[volume_col].values
            returns = np.diff(price, prepend=price[0])
            volatility = np.abs(returns)
            
            # Correlation between volume and volatility
            vol_volatility_corr = np.corrcoef(volume[1:], volatility[1:])[0, 1]
            
            # Volume distribution
            volume_dist = T._fit_distribution(volume)
            
            # GARCH-like effects: volume autocorrelation
            volume_acf = T._autocorrelation(volume, max_lag=20)
            
            results['volume_params'] = {
                'mean': float(np.mean(volume)),
                'std': float(np.std(volume)),
                'distribution': volume_dist,
                'volatility_correlation': float(vol_volatility_corr),
                'autocorrelation': volume_acf.tolist(),
                'garch_alpha': float(np.abs(vol_volatility_corr)),  # Simplified
                'mean_reversion': float(1 - volume_acf[1]) if len(volume_acf) > 1 else 0.1
            }
        
        # ===== 6. Spread Analysis =====
        if bid_col and ask_col:
            bid = df[bid_col].values
            ask = df[ask_col].values
            spread = ask - bid
            spread_pct = spread / price
            
            spread_dist = T._fit_distribution(spread_pct)
            
            # Correlation with volatility
            spread_volatility_corr = np.corrcoef(spread_pct[1:], volatility[1:])[0, 1]
            
            results['spread_params'] = {
                'mean_absolute': float(np.mean(spread)),
                'mean_percentage': float(np.mean(spread_pct)),
                'std_percentage': float(np.std(spread_pct)),
                'distribution': spread_dist,
                'volatility_correlation': float(spread_volatility_corr)
            }
        
        # ===== 7. Data Quality Analysis =====
        quality = T.analyze_data_quality(df, price_col, volume_col, time_col, 
                                         bid_col, ask_col)
        results['quality_report'] = quality
        
        return results
    
    @staticmethod
    def _fit_distribution(data):
        """
        Fit data to various distributions and return best fit
        """
        from scipy import stats
        
        distributions = {
            'normal': stats.norm,
            'lognormal': stats.lognorm,
            'gamma': stats.gamma,
            'exponential': stats.expon,
            'weibull': stats.weibull_min
        }
        
        best_dist = None
        best_ks_stat = np.inf
        best_params = None
        
        for name, dist in distributions.items():
            try:
                params = dist.fit(data)
                ks_stat, p_value = stats.kstest(data, lambda x: dist.cdf(x, *params))
                
                if ks_stat < best_ks_stat:
                    best_ks_stat = ks_stat
                    best_dist = name
                    best_params = params
            except:
                continue
        
        return {
            'type': best_dist,
            'parameters': best_params,
            'ks_statistic': float(best_ks_stat),
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'skewness': float(stats.skew(data)),
            'kurtosis': float(stats.kurtosis(data))
        }
    
    @staticmethod
    def _autocorrelation(data, max_lag=20):
        """
        Calculate autocorrelation function
        """
        data = data - np.mean(data)
        acf = np.correlate(data, data, mode='full')
        acf = acf[len(acf)//2:]
        acf = acf / acf[0]
        return acf[:max_lag]
    
    @staticmethod
    def analyze_data_quality(df, price_col='price', volume_col='volume', 
                            time_col='time', bid_col=None, ask_col=None,
                            expected_interval_seconds=1.0):
        """
        Comprehensive data quality analysis to detect:
        - Missing timestamps (gaps in data)
        - High anomaly time inconsistency (connection issues)
        - Price/volume outliers
        - Suspicious patterns
        - Data integrity issues
        
        Parameters:
        -----------
        df : pd.DataFrame
            Market data
        expected_interval_seconds : float
            Expected time between samples (e.g., 1.0 for 1-second bars)
            
        Returns:
        --------
        dict : Quality report with issues and recommendations
        """
        from scipy import stats
        
        report = {
            'total_records': len(df),
            'timestamp_issues': {},
            'price_issues': {},
            'volume_issues': {},
            'spread_issues': {},
            'overall_quality': 'GOOD',
            'warnings': [],
            'errors': []
        }
        
        # ===== 1. Timestamp Analysis =====
        if time_col in df.columns:
            time_data = pd.to_datetime(df[time_col], errors='coerce')
            
            # Calculate time differences
            time_diffs = time_data.diff().dt.total_seconds()
            
            # Detect gaps
            expected_diff = expected_interval_seconds
            large_gaps = time_diffs[time_diffs > expected_diff * 10]
            
            # Detect irregular intervals (jitter)
            if len(time_diffs) > 1:
                interval_std = np.std(time_diffs.dropna())
                interval_mean = np.mean(time_diffs.dropna())
                coefficient_of_variation = interval_std / interval_mean if interval_mean > 0 else np.inf
                
                # High CV indicates connection issues
                timestamp_quality = 'GOOD'
                if coefficient_of_variation > 1.0:
                    timestamp_quality = 'POOR'
                    report['errors'].append(
                        f"High timestamp irregularity (CV={coefficient_of_variation:.2f}). "
                        "Likely connection issues or broker feed problems."
                    )
                elif coefficient_of_variation > 0.5:
                    timestamp_quality = 'FAIR'
                    report['warnings'].append(
                        f"Moderate timestamp irregularity (CV={coefficient_of_variation:.2f}). "
                        "Possible intermittent connection issues."
                    )
                
                report['timestamp_issues'] = {
                    'mean_interval': float(interval_mean),
                    'std_interval': float(interval_std),
                    'coefficient_of_variation': float(coefficient_of_variation),
                    'num_large_gaps': int(len(large_gaps)),
                    'largest_gap_seconds': float(time_diffs.max()) if len(time_diffs) > 0 else 0,
                    'quality': timestamp_quality,
                    'missing_periods': large_gaps.index.tolist() if len(large_gaps) > 0 else []
                }
                
                # Detect duplicate timestamps
                duplicates = time_data.duplicated().sum()
                if duplicates > 0:
                    report['timestamp_issues']['duplicate_timestamps'] = int(duplicates)
                    report['warnings'].append(f"Found {duplicates} duplicate timestamps")
        
        # ===== 2. Price Analysis =====
        price = df[price_col].values
        
        # Outlier detection using Z-score and IQR
        price_zscore = np.abs(stats.zscore(price))
        price_outliers_zscore = np.sum(price_zscore > 3)
        
        q1, q3 = np.percentile(price, [25, 75])
        iqr = q3 - q1
        price_outliers_iqr = np.sum((price < q1 - 3*iqr) | (price > q3 + 3*iqr))
        
        # Detect flash crashes (sudden large moves)
        returns = np.diff(price) / price[:-1]
        flash_crashes = np.sum(np.abs(returns) > 0.05)  # >5% moves
        
        # Check for zero or negative prices
        invalid_prices = np.sum(price <= 0)
        
        # Check for constant prices (stale data)
        price_changes = np.diff(price)
        stale_periods = T._find_constant_periods(price, min_length=10)
        
        price_quality = 'GOOD'
        if invalid_prices > 0:
            price_quality = 'CRITICAL'
            report['errors'].append(f"Found {invalid_prices} zero or negative prices - DATA CORRUPTED")
        elif price_outliers_iqr > len(price) * 0.01:  # >1% outliers
            price_quality = 'POOR'
            report['warnings'].append(f"High number of price outliers ({price_outliers_iqr})")
        
        report['price_issues'] = {
            'outliers_zscore': int(price_outliers_zscore),
            'outliers_iqr': int(price_outliers_iqr),
            'flash_crashes': int(flash_crashes),
            'invalid_prices': int(invalid_prices),
            'stale_periods': len(stale_periods),
            'max_absolute_return': float(np.max(np.abs(returns))) if len(returns) > 0 else 0,
            'quality': price_quality
        }
        
        if len(stale_periods) > 0:
            report['warnings'].append(
                f"Found {len(stale_periods)} periods with constant prices (stale data)"
            )
        
        # ===== 3. Volume Analysis =====
        if volume_col in df.columns:
            volume = df[volume_col].values
            
            # Check for zeros
            zero_volume = np.sum(volume == 0)
            
            # Outlier detection
            volume_zscore = np.abs(stats.zscore(volume[volume > 0]))
            volume_outliers = np.sum(volume_zscore > 4)
            
            # Check for negative volumes
            negative_volume = np.sum(volume < 0)
            
            volume_quality = 'GOOD'
            if negative_volume > 0:
                volume_quality = 'CRITICAL'
                report['errors'].append(f"Found {negative_volume} negative volumes - DATA CORRUPTED")
            elif zero_volume > len(volume) * 0.1:  # >10% zeros
                volume_quality = 'POOR'
                report['warnings'].append(f"High number of zero volumes ({zero_volume})")
            
            report['volume_issues'] = {
                'zero_volume': int(zero_volume),
                'negative_volume': int(negative_volume),
                'outliers': int(volume_outliers),
                'quality': volume_quality
            }
        
        # ===== 4. Spread Analysis (if bid/ask available) =====
        if bid_col and ask_col:
            bid = df[bid_col].values
            ask = df[ask_col].values
            spread = ask - bid
            
            # Check for negative spreads (crossed book)
            negative_spreads = np.sum(spread < 0)
            
            # Check for unreasonably wide spreads
            spread_pct = spread / ((bid + ask) / 2)
            wide_spreads = np.sum(spread_pct > 0.01)  # >1%
            
            # Check for zero spreads
            zero_spreads = np.sum(spread == 0)
            
            spread_quality = 'GOOD'
            if negative_spreads > 0:
                spread_quality = 'CRITICAL'
                report['errors'].append(
                    f"Found {negative_spreads} negative spreads (crossed book) - DATA ISSUE"
                )
            elif wide_spreads > len(spread) * 0.05:
                spread_quality = 'FAIR'
                report['warnings'].append(f"Found {wide_spreads} unusually wide spreads")
            
            report['spread_issues'] = {
                'negative_spreads': int(negative_spreads),
                'zero_spreads': int(zero_spreads),
                'wide_spreads': int(wide_spreads),
                'mean_spread_pct': float(np.mean(spread_pct)),
                'quality': spread_quality
            }
        
        # ===== 5. Overall Quality Assessment =====
        quality_scores = []
        
        if 'quality' in report['timestamp_issues']:
            quality_scores.append(report['timestamp_issues']['quality'])
        quality_scores.append(report['price_issues']['quality'])
        if 'quality' in report['volume_issues']:
            quality_scores.append(report['volume_issues']['quality'])
        if 'quality' in report['spread_issues']:
            quality_scores.append(report['spread_issues']['quality'])
        
        if 'CRITICAL' in quality_scores:
            report['overall_quality'] = 'CRITICAL'
        elif 'POOR' in quality_scores:
            report['overall_quality'] = 'POOR'
        elif 'FAIR' in quality_scores:
            report['overall_quality'] = 'FAIR'
        else:
            report['overall_quality'] = 'GOOD'
        
        # Add recommendations
        report['recommendations'] = []
        if report['overall_quality'] in ['POOR', 'CRITICAL']:
            report['recommendations'].append("❌ Data quality issues detected. Clean data before use.")
        if len(report['timestamp_issues'].get('missing_periods', [])) > 0:
            report['recommendations'].append("⚠️ Fill missing timestamps with interpolation or forward fill")
        if report['price_issues'].get('stale_periods', 0) > 0:
            report['recommendations'].append("⚠️ Remove or interpolate stale price periods")
        if report['overall_quality'] == 'GOOD':
            report['recommendations'].append("✅ Data quality is good. Safe to use for analysis.")
        
        return report
    
    @staticmethod
    def _find_constant_periods(data, min_length=5):
        """
        Find periods where data remains constant (stale data detection)
        """
        diffs = np.diff(data)
        is_constant = (diffs == 0)
        
        periods = []
        start = None
        length = 0
        
        for i, const in enumerate(is_constant):
            if const:
                if start is None:
                    start = i
                length += 1
            else:
                if start is not None and length >= min_length:
                    periods.append({'start': start, 'end': start + length, 'length': length})
                start = None
                length = 0
        
        # Check last period
        if start is not None and length >= min_length:
            periods.append({'start': start, 'end': start + length, 'length': length})
        
        return periods
    
    def replicate_from_real_data(self, fit_results):
        """
        Configure this T instance to replicate characteristics from real data.
        
        Parameters:
        -----------
        fit_results : dict
            Output from fit_from_timeseries()
            
        Returns:
        --------
        self : T instance configured to match real data
        """
        # Reset with correct length
        self.tlength = fit_results['length']
        self.x = np.arange(self.tlength, dtype=int)
        
        # Reconstruct trend
        if 'polynomial' in fit_results['trend_params']:
            poly = fit_results['trend_params']['polynomial']
            coeffs = np.array(poly['coefficients'])
            trend = np.polyval(coeffs, self.x)
        else:
            trend = np.zeros(self.tlength)
        
        # Add sinusoidal components
        signal_component = np.zeros(self.tlength)
        if 'sinusoidal' in fit_results['trend_params']:
            for comp in fit_results['trend_params']['sinusoidal'][:3]:  # Top 3
                signal_component += comp['amplitude'] * np.sin(
                    2 * np.pi * comp['frequency'] * self.x + comp['phase']
                )
        
        # Add noise
        noise_params = fit_results['noise_params']
        noise = np.random.randn(self.tlength) * noise_params['std']
        
        # Combine all components
        self.Tsin = trend + signal_component + noise
        self.price = self.Tsin.copy()
        
        # Normalize to match real data range
        price_stats = fit_results['price_stats']
        self.price = (self.price - np.mean(self.price)) / np.std(self.price)
        self.price = self.price * price_stats['std'] + price_stats['mean']
        self.Tsin = self.price.copy()
        
        # Add volume if parameters available
        if 'volume_params' in fit_results and fit_results['volume_params']:
            vol_params = fit_results['volume_params']
            self.add_volume_state_space(
                base_volume=vol_params['mean'],
                volatility_impact=vol_params.get('volatility_correlation', 0.5)
            )
        
        # Add spread if parameters available
        if 'spread_params' in fit_results and fit_results['spread_params']:
            spread_params = fit_results['spread_params']
            self.add_bid_ask_spread(
                base_spread=spread_params['mean_percentage'],
                distribution=spread_params['distribution']['type'],
                sigma=spread_params['std_percentage']
            )
        
        self._update_dataframe()
        
        return self


# %% Example usage
# Create synthetic market data
t = T()
t.add_noise(0.3)
t.add_sinSerie(0.002, 2)  # Add trend
t.add_volume_state_space()  # Generate correlated volume

# Access multivariate data
df = t.df  # Contains: T, price, volume, volatility
print(df.head())

# Or try different volume models
t.add_volume_regime_switching()  # Switch to HMM
t.add_volume_microstructure()    # Or microstructure model
# %%
# Create synthetic market with realistic microstructure
t = T()
t.add_noise(0.5)
t.add_sinSerie(0.002, 2)  # Add trend

# Add volume
t.add_volume_state_space(base_volume=1000)

# Add spread (choose one):
# 1. Simple with distribution
t.add_bid_ask_spread(base_spread=0.01, distribution='lognormal', sigma=0.3)

# OR 2. Order flow based
t.add_bid_ask_order_flow(base_spread=0.01, imbalance_impact=0.5)

# OR 3. Inventory based
t.add_bid_ask_inventory(base_spread=0.01, inventory_impact=0.3)

# Access full market data
df = t.df
print(df.columns)  # T, price, volume, bid, ask, spread, ...
# %%
# Generate full synthetic market
t = T()
t.add_noise(0.5)
t.add_sinSerie(0.002, 2)
t.add_volume_state_space(base_volume=1000)
t.add_bid_ask_spread(base_spread=0.01, distribution='lognormal', sigma=0.3)

# Quick review of all data
t.plot_overview()

# Zoom into spread dynamics
t.plot_market_depth(window=200)

# Check correlations
t.plot_correlation_matrix()

# Or use pandas directly
t.df[['price', 'volume']].plot(subplots=True, figsize=(12, 6))
# %%

from assistant import T

# Extract everything from real BTC data
fit_results = T.fit_from_timeseries(
    filepath='C:/_projects/Lab/GDA/hd/merged/BTCUSD_20251231.csv',
    price_col='bid',
    volume_col='volume',
    time_col='time',
    bid_col='bid',
    ask_col='ask'
)

# Check data quality
quality = fit_results['quality_report']
print(f"Overall Quality: {quality['overall_quality']}")

# Check for connection issues
# if quality['timestamp_issues']['coefficient_of_variation'] > 1.0:
#     print("⚠️ BROKER CONNECTION ISSUES DETECTED")

# Generate matching synthetic data
t = T()
t.replicate_from_real_data(fit_results)
t.plot_overview()
# %%

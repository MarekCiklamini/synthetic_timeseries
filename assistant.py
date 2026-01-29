# -*- coding: utf-8 -*-
"""
"""
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
    
    def _update_dataframe(self):
        """Update dataframe with all available series"""
        data = {'T': self.x, 'price': self.price}
        
        if self.volume is not None:
            data['volume'] = self.volume
        
        if self.volatility_state is not None:
            data['volatility'] = self.volatility_state
        
        if hasattr(self, 'regime'):
            data['regime'] = self.regime
        
        # Keep backward compatibility
        data['y'] = self.Tsin
        
        self.df = pd.DataFrame(data)

class results_compilation(object):
    
    def __init__(self,): 
        import pandas as pd
        import glob
    # def add_rorces()
    def strContainsRewriteAt(self,strList2rewrite):
        df = self.D
        o = df['ID'].str.contains(strList2rewrite[0])
        df['ID'].loc[o]  = strList2rewrite[1]
    def dataset_cleaning_test(self, RF_DataFrame):
        '''
        
        Returns
        -------
        TYPE
            DESCRIPTION.
        Comment:
            https://thispointer.com/pandas-select-dataframe-columns-containing-string/
         
         
        '''
        
        
        df = RF_DataFrame
        sL = []
        sL.append(['Force Reaction Contact Bracket Fixture: '     ,   'M8 R'] )   
        sL.append(['Force Reaction Contact Bracket Fixture 1: '   ,   'M8 R' ]) 
        sL.append(['Force Reaction Contact Bracket Fixture 2: '   ,   'M8 L' ]) 
        sL.append(['Force Reaction Contact Bracket Block: '       ,   'M6 R' ])
        sL.append(['Force Reaction Contact Bracket Block 1: '     ,   'M6 R' ])
        sL.append(['Force Reaction Contact Bracket Block 2: '     ,   'M6 L' ])

        for s_ in sL:            self.strContainsRewriteAt(s_)
        df['AX']  = df['AX'].abs()
            
            
    def dataset_cleaning(self, RF_DataFrame):
        '''
        
        Returns
        -------
        TYPE
            DESCRIPTION.
        Comment:
            https://thispointer.com/pandas-select-dataframe-columns-containing-string/
        '''
        
        df = RF_DataFrame
        sM8R  = 'Force Reaction Contact Bracket Fixture 2: '
        
        # o = df['ID'].str.contains('Fixture 2')
        o = df['ID'].str.equal('Fixture 2')
        df['ID'].loc[o]  = 'M8 Right'
        o = df['ID'].str.contains('Fixture:')
        df['ID'].loc[o]  = 'M8 Left'
        
        o = df['ID'].str.contains('Bracket Block 2: ')
        df['ID'].loc[o]  = 'M6 Right'
        o = df['ID'].str.contains('Bracket Block: ')
        df['ID'].loc[o]  = 'M6 Left'
        # self.D = df
        df['AX']  = df['AX'].abs()
        return df
    def getDataframeFromResults(self, path2collection):
        import pandas as pd
        '''
        
        Parameters
        ----------
        path2collection : TYPE
            DESCRIPTION.
    
        Returns
        -------
        None.
        
        Sources
        -------
        https://www.datasciencemadesimple.com/square-root-of-the-column-in-pandas-python-2/
        https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html
        https://stackoverflow.com/questions/10373660/converting-a-pandas-groupby-output-from-series-to-dataframe
        https://stackoverflow.com/questions/43967663/scatter-plot-with-legend-colored-by-group-without-multiple-calls-to-plt-scatter
    
        '''
        content = glob.glob(path2collection + '/*')
    
        D = pd.DataFrame()
        for f in content:
            iid = f.split('\\')[-1].split('_')
            lc = iid[0]     # loadcase
            ca = iid[1][-1]      # component axis 
            
            d = pd.read_csv(f, names=['ID', 'X', 'Y','Z'])
            
            d['LC'] = lc
            d['CA'] = ca
            d['AX'] = d['Y']
            D = pd.concat([D,d])
        u = D.loc[D['X']==0].index
        D = D.drop(u)                   # vyhodit nuly
        D['SH'] = (D['X'] ** 2 + D['Z'] ** 2)  ** (1/2)
        D = D.sort_values(by=['SH'])
        self.D = D
        return self.D
    
class api_results(object):
    ''' need daframe by class results_compilation
    '''
    def __init__(self,ResultsForceDataFrame):
    
        self.df = ResultsForceDataFrame
        
    def app_scatter_graph_2(self, ):
        size_marker = 15
        D = self.df
        app = dash.Dash(__name__)
        fig1 = px.line(D,  )
        # fig2 = px.scatter(D, x="SH", y="LC_CA", color="ID", )
        
        app.layout = html.Div(children=[
            html.H1(children='Bolts - Reaction forces'),
        
            html.Div(children='''
                Bolt forces compilation from simulation 
            '''),
        
            dcc.Graph(
                id='graph Axial',
                figure=fig1
            ),
                
            # dcc.Graph(id='graph Shear',figure=fig2)
        ])
        fig1.update_traces(marker=dict(size=size_marker))
        # fig2.update_traces(marker=dict(size=size_marker))
        # fig1.write_html("axial_forces.html")
        # fig2.write_html("shear_forces.html")
        # fig1.show()
        # fig2.show()
        return app          
    def app_scatter_graph(self, ):
        D = self.df
        app = dash.Dash(__name__)
        fig1 = px.scatter(D, x="AX", y="LC", color="ID", )
        fig2 = px.scatter(D, x="SH", y="LC", color="ID", )
        
        app.layout = html.Div(children=[
            html.H1(children='Bolts - Reaction forces'),
        
            html.Div(children='''
                Dash: A web application framework for your data.
            '''),
        
            dcc.Graph(
                id='graph Axial',
                figure=fig1
            ),
                
            dcc.Graph(
                id='graph Shear',
                figure=fig2
            )
        ])
        return app      
    def app_bar_graph(self, ):
        D = self.df
        app = dash.Dash(__name__)
        fig1 = px.bar(D, x="AX", y="LC", color="ID", barmode="group")
        fig2 = px.bar(D, x="SH", y="LC", color="ID", barmode="group")
        
        app.layout = html.Div(children=[
            html.H1(children='Bolts - Reaction forces'),
        
            html.Div(children='''
                Dash: A web application framework for your data.
            '''),
        
            dcc.Graph(
                id='graph Axial',
                figure=fig1
            ),
                
            dcc.Graph(
                id='graph Shear',
                figure=fig2
            )
        ])
        return app
       

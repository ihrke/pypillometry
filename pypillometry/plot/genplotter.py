import itertools
from typing import Iterable, Optional, Tuple
from ..eyedata import GenericEyeData
import numpy as np
from loguru import logger

import pylab as plt
import matplotlib.patches as patches
from matplotlib import cm
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages

class GenericPlotter:
    """
    Abstract Class for plotting eye data. This class is not meant to be used directly.
    """
    obj: GenericEyeData # link to the data object

    def plot_intervals(self, intervals: list|np.ndarray,
                       eyes: str|list=[], variables: str|list=[],
                       pdf_file: Optional[str]=None, nrow: int=5, ncol: int=3, 
                       figsize: Tuple[int,int]=(10,10), 
                       units: str="ms", 
                       plot_index: bool=True):
        """"
        Plotting data around intervals.

        Each interval gets a separate subplot. 
        The data is plotted for each eye and variable in different colors.
        
        Parameters
        ----------
        eyes: str or list
            eyes to plot
        variables: str or list
            variables to plot
        intervals: list of tuples
            intervals to plot
        pdf_file: str or None
            if the name of a file is given, the figures are saved into a 
            multi-page PDF file
        ncol: int
            number of columns for the subplots for the intervals
        nrow: int
            number of rows for the subplots for the intervals
        units: str
            units in which the signal is plotted
        plot_index: bool
            plot a number with the blinks' index (e.g., for identifying abnormal blinks)
            
        """
        obj=self.obj # PupilData object
        fac=obj._unit_fac(units)
        nsubplots=nrow*ncol # number of subplots per figure

        eyes,variables=obj._get_eye_var(eyes, variables)
        if isinstance(intervals, np.ndarray):
            if intervals.ndim!=2 or intervals.shape[1]!=2:
                raise ValueError("intervals must be a list of tuples or a 2D array with 2 columns")
            intervals=intervals.tolist()
        if isinstance(intervals, Iterable):
            intervals=[tuple(i) for i in intervals]
            
        nfig=int(np.ceil(len(intervals)/nsubplots))

        figs=[]
        if isinstance(pdf_file,str):
            _backend=mpl.get_backend()
            mpl.use("pdf")
            plt.ioff() ## avoid showing plots when saving to PDF 
        
        iinterv=0
        for i in range(nfig):
            fig=plt.figure(figsize=figsize)
            axs = fig.subplots(nrow, ncol).flatten()

            for ix,(start,end) in enumerate(intervals[(i*nsubplots):(i+1)*nsubplots]):
                iinterv+=1
                slic=slice(start,end)
                ax=axs[ix]
                for eye,var in itertools.product(eyes,variables):
                    ax.plot(obj.tx[slic]*fac,obj.data[eye,var][slic], label="%s_%s"%(eye,var))

                if plot_index: 
                    ax.text(0.5, 0.5, '%i'%(iinterv), fontsize=12, horizontalalignment='center',     
                            verticalalignment='center', transform=ax.transAxes)
            figs.append(fig)

        if pdf_file is not None:
            print("> Saving file '%s'"%pdf_file)
            with PdfPages(pdf_file) as pdf:
                for fig in figs:
                    pdf.savefig(fig)
            ## switch back to original backend and interactive mode                
            mpl.use(_backend) 
            plt.ion()
            
        return figs    
        
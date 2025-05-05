from ..eyedata import GenericEyeData
from .genplotter import GenericPlotter
import numpy as np
from collections.abc import Iterable

import pylab as plt
import matplotlib.patches as patches
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages

class GazePlotter(GenericPlotter):
    """
    Class for plotting eye data. The class is initialized with an EyeData object
    and provides methods to plot the data in various ways.
    """
    obj: GenericEyeData # link to the data object

    def __init__(self, obj: GenericEyeData):
        self.ignore_vars = [] #["blinkmask", "pupilinterpolated"]
        self.obj = obj
    
    def plot_heatmap(self, 
            plot_range: tuple=(-np.inf, +np.inf), 
            eyes: list=[],
            plot_screen: bool=True,
            units: str="sec",
            figsize: tuple=(10,10),
            cmap: str="jet",
            gridsize="auto"
            ) -> None:
        """
        Plot EyeData as a heatmap. Typically used for a large amount of data
        to spot the most frequent locations.
        To plot a scanpath, use :meth:`.plot_scanpath()` instead.
        Each eye is plotted in a separate subplot.

        Parameters:
        -----------

        plot_range: tuple
            The time range to plot. Default is (-np.inf, +np.inf), i.e. all data.
        eyes: list
            The eyes to plot. Default is [], which means all available data ("left", "right",
            average, regression, ...)
        plot_screen: bool
            Whether to plot the screen limits. Default is True.
        units: str
            The units to plot. Default is "sec".
        figsize: tuple
            The figure size (per subplot). Default is (10,5).
        cmap: str
            The colormap to use. Default is "jet".
        gridsize: str or int
            The gridsize for the hexbin plot. Default is "auto".
        """
        obj = self.obj
        
        fac=obj._unit_fac(units)
        tx=obj.tx*fac

        # plot_range
        start,end=plot_range
        if start==-np.inf:
            startix=0
        else:
            startix=np.argmin(np.abs(tx-start))
            
        if end==np.inf:
            endix=tx.size
        else:
            endix=np.argmin(np.abs(tx-end))    
        
        # which eyes to plot
        if len(eyes)==0:
            eyes=obj.eyes
        if not isinstance(eyes, list):
            plot_eyes=[eyes]

        # choose gridsize
        if gridsize=="auto":
            gridsize=int(np.sqrt(endix-startix))

        nplots = len(eyes)
        fig, axs = plt.subplots(1,nplots)
        # for the case when nplots=1, make axs iterable
        if not isinstance(axs, Iterable):
            axs=[axs]
        fig.set_figheight(figsize[1])
        fig.set_figwidth(figsize[0]*nplots)
        for eye,ax in zip(eyes, axs):
            vx = "_".join([eye,"x"])
            vy = "_".join([eye,"y"])
            if vx in obj.data.keys() and vy in obj.data.keys():
                divider = make_axes_locatable(ax)
                im=ax.hexbin(obj.data[vx][startix:endix], obj.data[vy][startix:endix], 
                            gridsize=gridsize, cmap=cmap, mincnt=1)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im, cax=cax, orientation='vertical')
            ax.set_title(eye)
            ax.set_aspect("equal")
            #fig.colorbar()
            if plot_screen:
                screenrect=patches.Rectangle((obj.screen_xlim[0], obj.screen_ylim[0]), 
                                obj.screen_xlim[1], obj.screen_ylim[1], 
                                fill=False, edgecolor="red", linewidth=2)
                ax.add_patch(screenrect)

    def plot_scanpath(self, 
            plot_range: tuple=(-np.inf, +np.inf), 
            eyes: list=[],
            plot_screen: bool=True,
            plot_onsets: bool=True,
            title: str="",
            units: str=None,
            figsize: tuple=(10,10)
            ) -> None:
        """
        Plot EyeData as a scanpath. Typically used for single trials
        or another small amount of data. To plot a larger amount of data,
        use :meth:`.plot_heatmap()` instead.

        If several eyes are available, they are plotted in the same figure 
        in different colors. You can choose between them using the `plot_eyes` 
        parameter.

        Parameters:
        -----------

        plot_range: tuple
            The time range to plot. Default is (-np.inf, +np.inf), i.e. all data.
        eyes: list
            The eyes to plot. Default is [], which means all available data ("left", "right",
            average, regression, ...)
        plot_screen: bool
            Whether to plot the screen limits. Default is True.
        plot_onsets: bool
            Whether to plot the event onsets. Default is True.
        title: str
            The title of the plot. Default is "".
        units: str
            The units to plot. Default is "sec". Plotted only for the first "eye".
            If None, the units are taken from the data.
        figsize: tuple
            The figure size (per subplot). Default is (10,5).
        """
        obj = self.obj

        if units is not None: 
            fac=obj._unit_fac(units)
            tx = obj.tx*fac
            evon = obj.event_onsets*fac
        else:
            tx=obj.tx.copy()
            evon=obj.event_onsets.copy()

        # plot_range
        start,end=plot_range
        if start==-np.inf:
            startix=0
        else:
            startix=np.argmin(np.abs(tx-start))
            
        if end==np.inf:
            endix=tx.size
        else:
            endix=np.argmin(np.abs(tx-end))    
        
        # get events in range
        evonixx=np.logical_and(evon>=start, evon<end)
        evlab=obj.event_labels[evonixx]
        evon=evon[evonixx]
        evontix=np.argmin(np.abs(tx-evon[:,np.newaxis]), axis=1).astype(int)

        # which eyes to plot
        if len(eyes)==0:
            eyes=obj.eyes

        fig, ax = plt.subplots(1,1)
        fig.set_figheight(figsize[1])
        fig.set_figwidth(figsize[0])

        tnorm = tx[startix:endix].copy()
        tnorm = (tnorm - tnorm.min()) / (tnorm.max() - tnorm.min())
        for eye in eyes:
            vx = "_".join([eye,"x"])
            vy = "_".join([eye,"y"])
            if vx in obj.data.keys() and vy in obj.data.keys():
                ax.plot(obj.data[vx][startix:endix], obj.data[vy][startix:endix], alpha=0.3, label=eye)
                ax.scatter(obj.data[vx][startix:endix], obj.data[vy][startix:endix], 
                        s=1, c=cm.jet(tnorm))
            if plot_onsets and eye==eyes[0]:
                for ix, lab in zip(evontix, evlab):
                    ax.text(obj.data[vx][ix], obj.data[vy][ix], lab, 
                            fontsize=12, ha="center", va="center")

        ax.set_aspect("equal")
        #fig.colorbar()
        if plot_screen:
            screenrect=patches.Rectangle((obj.screen_xlim[0], obj.screen_ylim[0]), 
                            obj.screen_xlim[1], obj.screen_ylim[1], 
                            fill=False, edgecolor="red", linewidth=2)
            ax.add_patch(screenrect)
        ax.legend()
        ax.set_title(title)
        fig.tight_layout()    
import pylab as plt
import matplotlib.patches as patches
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

class GazePlotter:
    """
    Class for plotting eye data. The class is initialized with an EyeData object
    and provides methods to plot the data in various ways.
    """
    obj: GenericEyeData # link to the data object

    def __init__(self, obj: GenericEyeData):
        self.obj = obj
    
    def plot_timeseries(self, 
            plot_range: tuple=(-np.infty, +np.infty),
            plot_data: list=[], 
            plot_eyes: list=[],
            plot_onsets: str="line",
            units: str=None,
            figsize: tuple=(10,5)
            ) -> None:
        """
        Plot a part of the EyeData. Each data type is plotted in a separate subplot.

        Parameters:
        -----------

        plot_range: tuple
            The time range to plot. Default is (-np.infty, +np.infty), i.e. all data.
        plot_data: list
            The data to plot. Default is [], which means all available data will be plotted.
            Available data are ["x","y","pupil"] but can be extended by the user.
        plot_eyes: list
            The eyes to plot. Default is [], which means all available data ("left", "right",
            "average", "regression", ...)
        plot_onsets: str
            Whether to plot markers for the event onsets. One of "line" (vertical lines),
            "label" (text label), "both" (both lines and labels), or "none" (no markers).
        units: str
            The units to plot. Default is "sec". If None, use the units in the time vector.
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
            units="tx"

        xlab=units

        # plot_range
        start,end=plot_range
        if start==-np.infty:
            startix=0
        else:
            startix=np.argmin(np.abs(tx-start))
            
        if end==np.infty:
            endix=tx.size
        else:
            endix=np.argmin(np.abs(tx-end))
        
        # which data to plot
        if len(plot_data)==0:
            plot_data=obj.data.get_available_variables()

        # which eyes to plot
        if len(plot_eyes)==0:
            plot_eyes=obj.data.get_available_eyes()

        # how to plot onsets
        if plot_onsets=="line":
            ev_line=True
            ev_label=False
        elif plot_onsets=="label":
            ev_line=False
            ev_label=True
        elif plot_onsets=="both":
            ev_line=True
            ev_label=True
        elif plot_onsets=="none":
            ev_line=False
            ev_label=False
        else:
            raise ValueError("plot_onsets must be one of 'line', 'label', 'both', or 'none'")
        
        tx=tx[startix:endix]
        ixx=np.logical_and(evon>=start, evon<end)
        evlab=obj.event_labels[ixx]
        evon=evon[ixx]

        nplots = len(plot_data)
        fig, axs = plt.subplots(nplots,1)
        # for the case when nplots=1, make axs iterable
        if not isinstance(axs, Iterable):
            axs=[axs]
        fig.set_figheight(figsize[1]*nplots)
        fig.set_figwidth(figsize[0])
        for var,ax in zip(plot_data, axs):
            for eye in plot_eyes:
                vname = "_".join([eye,var])
                if vname in obj.data.keys():
                    ax.plot(tx, obj.data[vname][startix:endix], label=eye)
            if ev_line:
                ax.vlines(evon, *ax.get_ylim(), color="grey", alpha=0.5)
            if ev_label:
                ll,ul=ax.get_ylim()
                for ev,lab in zip(evon,evlab):
                    ax.text(ev, ll+(ul-ll)/2., "%s"%lab, fontsize=8, rotation=90)
            ax.set_title(var)
            ax.legend()
        
        plt.legend()
        plt.xlabel(xlab)        

    def plot_heatmap(self, 
            plot_range: tuple=(-np.infty, +np.infty), 
            plot_eyes: list=[],
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
            The time range to plot. Default is (-np.infty, +np.infty), i.e. all data.
        plot_eyes: list
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
        if start==-np.infty:
            startix=0
        else:
            startix=np.argmin(np.abs(tx-start))
            
        if end==np.infty:
            endix=tx.size
        else:
            endix=np.argmin(np.abs(tx-end))    
        
        # which eyes to plot
        if len(plot_eyes)==0:
            plot_eyes=obj.data.get_available_eyes()
        if not isinstance(plot_eyes, list):
            plot_eyes=[plot_eyes]

        # choose gridsize
        if gridsize=="auto":
            gridsize=int(np.sqrt(endix-startix))

        nplots = len(plot_eyes)
        fig, axs = plt.subplots(1,nplots)
        # for the case when nplots=1, make axs iterable
        if not isinstance(axs, Iterable):
            axs=[axs]
        fig.set_figheight(figsize[1])
        fig.set_figwidth(figsize[0]*nplots)
        for eye,ax in zip(plot_eyes, axs):
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
            plot_range: tuple=(-np.infty, +np.infty), 
            plot_eyes: list=[],
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
            The time range to plot. Default is (-np.infty, +np.infty), i.e. all data.
        plot_eyes: list
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
        if start==-np.infty:
            startix=0
        else:
            startix=np.argmin(np.abs(tx-start))
            
        if end==np.infty:
            endix=tx.size
        else:
            endix=np.argmin(np.abs(tx-end))    
        
        # get events in range
        evonixx=np.logical_and(evon>=start, evon<end)
        evlab=obj.event_labels[evonixx]
        evon=evon[evonixx]
        evontix=np.argmin(np.abs(tx-evon[:,np.newaxis]), axis=1).astype(int)

        # which eyes to plot
        if len(plot_eyes)==0:
            plot_eyes=obj.data.get_available_eyes()

        fig, ax = plt.subplots(1,1)
        fig.set_figheight(figsize[1])
        fig.set_figwidth(figsize[0])

        tnorm = tx[startix:endix].copy()
        tnorm = (tnorm - tnorm.min()) / (tnorm.max() - tnorm.min())
        for eye in plot_eyes:
            vx = "_".join([eye,"x"])
            vy = "_".join([eye,"y"])
            if vx in obj.data.keys() and vy in obj.data.keys():
                ax.plot(obj.data[vx][startix:endix], obj.data[vy][startix:endix], alpha=0.3, label=eye)
                ax.scatter(obj.data[vx][startix:endix], obj.data[vy][startix:endix], 
                        s=1, c=cm.jet(tnorm))
            if plot_onsets and eye==plot_eyes[0]:
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
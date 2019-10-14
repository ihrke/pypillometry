import numpy as np

def nprange(ar):
    return (ar.min(),ar.max())


def plot_pupil_ipy(tx, sy, event_onsets=None, overlays=None, overlay_labels=None, figsize=(16,8)):
    from ipywidgets import interact, interactive, fixed, interact_manual, Layout
    import ipywidgets as widgets

    def draw_plot(plotxrange):
        xmin,xmax=plotxrange
        ixmin=np.argmin(np.abs(tx-xmin))
        ixmax=np.argmin(np.abs(tx-xmax))
        plt.figure(figsize=figsize)

        plt.plot(tx[ixmin:ixmax],sy[ixmin:ixmax])
        if overlays is not None:
            if type(overlays) is np.ndarray:
                plt.plot(tx[ixmin:ixmax],overlays[ixmin:ixmax],label=overlay_labels)
            else:
                for i,overlay in enumerate(overlays):
                    lab=overlay_labels[i] if overlay_labels is not None else None
                    plt.plot(tx[ixmin:ixmax],overlay[ixmin:ixmax], label=lab)
        plt.vlines(event_onsets, *plt.ylim(), color="grey", alpha=0.5)
        plt.xlim(xmin,xmax)
        if overlay_labels is not None:
            plt.legend()


    wid_range=widgets.FloatRangeSlider(
        value=[tx.min(), tx.max()],
        min=tx.min(),
        max=tx.max(),
        step=1,
        description=' ',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',
        layout=Layout(width='100%', height='80px')
    )

    interact(draw_plot, plotxrange=wid_range)
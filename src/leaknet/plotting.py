"""
Plots: the live training plot for notebooks, a static version of it for saved histories, and views of a scene.
"""

from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from scipy.stats import multivariate_normal

from leaknet.simulation import concentration_field

FONT = "Lucida Console, monospace"


### training -----------------------------------------------------------------------------------------------------------
class TrainingPlotter:
    """
    Convenience class for adding an interactive training visualization to jupyter notebooks. Does not work with .py!

        plotter = TrainingPlotter(n_epochs, n_train_samples)
        train(data, ..., plotter=plotter)
    """

    def __init__(self, n_epochs, l_dataset, y_axis_range=(-0.1, 1.1), y_title="loss (distance to leak)", aux_names=()):
        from IPython.display import display

        # processing related stats
        self.samp_tota = n_epochs * l_dataset
        self.spd_deque = deque(maxlen=10)
        self.spd_avg = 0
        self.val_best = None

        # some design parameters
        self.padding_factor = 0.03
        self.y_axis_range = list(y_axis_range)
        self.y_title = y_title
        self.aux_names = aux_names

        # initialize the plot once: with all the dummy traces and to actually show the figure once
        self.fig_loss = self._initialize_loss_plots()
        display(self.fig_loss)

    def _initialize_loss_plots(self):
        fig_loss = go.FigureWidget()

        # add all the dummy traces so that the legend is displayed from the beginning on
        # trace [0] / [1] training loss variance band upper / lower
        fig_loss.add_scatter(x=[None], y=[None], mode="lines", name="upper_training", line=dict(color="firebrick", width=0), showlegend=False)
        fig_loss.add_scatter(x=[None], y=[None], mode="lines", name="train. stdv", line=dict(color="firebrick", width=0), fill="tonexty", fillcolor="rgba(178, 34, 34, 0.14)")
        # trace [2] / [3] validation loss variance band upper / lower
        fig_loss.add_scatter(x=[None], y=[None], mode="lines", name="valid_upper", line=dict(color="seagreen", width=0), showlegend=False)
        fig_loss.add_scatter(x=[None], y=[None], mode="lines", name="valid. stdv", line=dict(color="seagreen", width=0), fill="tonexty", fillcolor="rgba(46, 139, 87, 0.16)", visible="legendonly")
        # trace [4] training loss
        fig_loss.add_scatter(x=[None], y=[None], mode="lines", name="train. loss", line=dict(color="firebrick", width=2.0))
        # trace [5] horizontal line for current best validation loss
        fig_loss.add_scatter(x=[None], y=[None], mode="lines+markers", name="Best val. loss", line=dict(color="seagreen", width=2, dash="dot"), marker=dict(size=[0, 18], opacity=[0, 1], symbol=["circle", "diamond-wide"]), showlegend=False)
        # trace [6] / [7] validation loss (white background + line)
        fig_loss.add_scatter(x=[None], y=[None], mode="lines", name="valid. loss", line=dict(color="white", width=3.0), showlegend=False, legendgroup="group valid")
        fig_loss.add_scatter(x=[None], y=[None], mode="lines", name="valid. loss", line=dict(color="seagreen", width=2.0), legendgroup="group valid")
        # trace [8...] auxiliary losses (e.g. TP / FN / FP parts of the sinkhorn loss)
        for name, color in zip(self.aux_names, ("darkorange", "burlywood", "turquoise")):
            fig_loss.add_scatter(x=[None], y=[None], mode="lines", name=f"aux. - {name}", line=dict(color=color, width=1.5))

        # layout settings
        axis_style = dict(
            title_standoff=30, title_font_color="grey", title_font_family=FONT, title_font_size=20,
            zeroline=True, zerolinewidth=3, zerolinecolor="gray", showgrid=True, gridwidth=1.5, gridcolor="lightgray",
            tickfont=dict(family=FONT, size=12, color="grey"),
        )  # fmt: skip
        fig_loss.update_layout(height=600, plot_bgcolor="white", margin_pad=10, legend=dict(font=dict(family=FONT, size=14, color="grey")))
        fig_loss.update_xaxes(
            title="number of samples seen (running total over all epochs)", griddash="dash",
            range=[-self.padding_factor * self.samp_tota, (1 + self.padding_factor) * self.samp_tota], **axis_style,
        )  # fmt: skip
        fig_loss.update_yaxes(title=self.y_title, range=self.y_axis_range, **axis_style)

        return fig_loss

    def _annotations(self):
        annotations = [
            # annotation for the processing speed
            dict(
                x=0.5, y=1.1, xref="paper", yref="paper", showarrow=False, xanchor="center",
                text=f"Samples/sec (moving avg): <b>{self.spd_avg:.2f}</b>",
                font=dict(family=FONT, size=16, color="gray"),
            )  # fmt: skip
        ]
        if self.val_best is not None:
            # annotation for the current best validation loss
            annotations.append(
                dict(
                    x=(1 + self.padding_factor) * self.samp_tota, y=self.val_best, xref="x", yref="y", showarrow=False,
                    text=f"<b> val. best <br> {self.val_best:09.4f}</b>",
                    font=dict(family=FONT, size=14, color="seagreen"), xanchor="left", yanchor="middle", align="left",
                )  # fmt: skip
            )
        return annotations

    def update_proc_speed(self, actual_bs, start_time, end_time):
        # append the current batch speed (samples / second) and calculate the new moving average
        self.spd_deque.append(actual_bs / max(end_time - start_time, 1e-9))
        self.spd_avg = sum(self.spd_deque) / len(self.spd_deque)

    def update_val_err_plot(self, val_avg, val_std_hi, val_std_lo):
        samp_seen = 0 if self.fig_loss.data[4].x[-1] is None else self.fig_loss.data[4].x[-1]
        self.val_best = val_avg if self.val_best is None else min(self.val_best, val_avg)

        with self.fig_loss.batch_update():
            # validation loss (background + trace)
            for idx in (6, 7):
                self.fig_loss.data[idx].x += (samp_seen,)
                self.fig_loss.data[idx].y += (val_avg,)
            # horizontal line indicator of best validation loss
            self.fig_loss.data[5].x = [0, (1 + self.padding_factor) * self.samp_tota]
            self.fig_loss.data[5].y = [self.val_best, self.val_best]
            # validation loss variance upper / lower
            self.fig_loss.data[2].x += (samp_seen,)
            self.fig_loss.data[2].y += (val_avg + val_std_hi,)
            self.fig_loss.data[3].x += (samp_seen,)
            self.fig_loss.data[3].y += (val_avg - val_std_lo,)

            self.fig_loss.update_layout(annotations=self._annotations())

    def update_tra_err_plot(self, samp_seen, tra_avg, tra_std_hi, tra_std_lo, aux=()):
        with self.fig_loss.batch_update():
            # training loss
            self.fig_loss.data[4].x += (samp_seen,)
            self.fig_loss.data[4].y += (tra_avg,)
            # training loss variance upper / lower
            self.fig_loss.data[0].x += (samp_seen,)
            self.fig_loss.data[0].y += (tra_avg + tra_std_hi,)
            self.fig_loss.data[1].x += (samp_seen,)
            self.fig_loss.data[1].y += (tra_avg - tra_std_lo,)
            # auxiliary losses
            for idx, value in enumerate(aux):
                self.fig_loss.data[8 + idx].x += (samp_seen,)
                self.fig_loss.data[8 + idx].y += (value,)

            self.fig_loss.update_layout(annotations=self._annotations())


def _nan_array(values):
    return np.array([np.nan if value is None else value for value in values], dtype=float)


def plot_history(history, ax=None, title=None, y_range=(-0.1, 1.1), y_label="loss (distance to leak)", bands=True, smooth=1):
    """
    Static version of the live training plot from a saved history (see train.save_run / load_run).

    smooth: moving average window (in batches) for the training curve, 1 = raw
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    tra, val = history["train"], history["valid"]
    x_tra, y_tra = np.array(tra["samples_seen"]), _nan_array(tra["loss"])
    x_val, y_val = np.array(val["samples_seen"]), _nan_array(val["loss"])

    def moving_average(y):
        if smooth <= 1:
            return y
        kernel = np.ones(smooth) / smooth
        return np.convolve(np.pad(y, (smooth // 2, smooth - 1 - smooth // 2), mode="edge"), kernel, mode="valid")

    if bands:
        hi = np.nan_to_num(_nan_array(tra["std_hi"]))
        lo = np.nan_to_num(_nan_array(tra["std_lo"]))
        ax.fill_between(x_tra, moving_average(y_tra - lo), moving_average(y_tra + hi), color="firebrick", alpha=0.14, lw=0, label="train. stdv")
    ax.plot(x_tra, moving_average(y_tra), color="firebrick", lw=1.5, label="train. loss")

    for idx, (name, color) in enumerate(zip(history.get("aux_names", []), ("darkorange", "burlywood", "turquoise"))):
        ax.plot(x_tra, moving_average(np.array([el[idx] for el in tra["aux"]])), color=color, lw=1.2, label=f"aux. - {name}")

    ax.plot(x_val, y_val, color="white", lw=3.5)
    ax.plot(x_val, y_val, color="seagreen", lw=2, label="valid. loss")

    best = np.nanmin(y_val)
    ax.axhline(best, color="seagreen", ls=":", lw=1.5)
    ax.annotate(f"val. best\n{best:.4f}", xy=(1.005, best), xycoords=("axes fraction", "data"), color="seagreen", va="center", fontsize=9, family="monospace")

    ax.axhline(0, color="gray", lw=1.5)
    ax.set_ylim(*y_range)
    ax.set_xlabel("number of samples seen (running total over all epochs)", color="grey")
    ax.set_ylabel(y_label, color="grey")
    ax.grid(color="lightgray", ls="--", lw=0.8)
    ax.tick_params(colors="grey")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(frameon=False, labelcolor="grey", loc="upper right")
    if title:
        ax.set_title(title, color="dimgray", loc="left")

    return ax


### scenes -------------------------------------------------------------------------------------------------------------
def plot_scene(sample, t=0, ax=None, field=None, show_sources=True, title=None):
    """one timestep of a scene: concentration field, sources, leaks, sensors and the wind arrow"""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    x, y, Z = field if field is not None else concentration_field(sample)

    n_sensors, n_leaks = int(sample["n_sensors"]), int(sample["n_leaks"])
    ax.contourf(x, y, Z[t], levels=20, cmap="GnBu")
    if show_sources:
        pos = sample["pos_sources"][: int(sample["n_sources"])]
        ax.scatter(pos[:, 0], pos[:, 1], s=14, color="black", edgecolors="lightgray", lw=0.5, label=f"sources ({len(pos)})")
    ax.scatter(*sample["pos_leaks"][:n_leaks].T, s=110, marker="^", color="seagreen", edgecolors="white", label=f"leaks ({n_leaks})", zorder=3)
    ax.scatter(*sample["pos_sensors"][:n_sensors].T, s=90, marker="p", color="firebrick", edgecolors="white", label=f"sensors ({n_sensors})", zorder=3)

    # wind arrow from the center, length ~ wind speed
    speed, direc = sample["wind_speed"][t], sample["wind_direc"][t]
    ax.annotate("", xy=((speed + 0.2) * 0.8 * np.cos(direc), (speed + 0.2) * 0.8 * np.sin(direc)), xytext=(0, 0), arrowprops=dict(color="mediumorchid", width=2, headwidth=8))

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.set_title(title if title is not None else f"t = {t} | wind speed {speed:.2f}", fontsize=10, color="dimgray")

    return ax


def plot_wind(sample, axes=None):
    """wind speed and direction over time plus the resulting wind covariance blob"""
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(14, 3.5), gridspec_kw=dict(width_ratios=[1.4, 1.4, 1]))

    axes[0].plot(sample["time"], sample["wind_speed"], color="mediumorchid")
    axes[0].set(ylim=(0, 1), xlabel="timestep", ylabel="wind speed")
    axes[1].plot(sample["time"], sample["wind_direc"], ".", color="mediumorchid")
    axes[1].set(ylim=(0, 2 * np.pi), xlabel="timestep", ylabel="wind direction [rad]")

    x = np.linspace(-1.2, 1.2, 100)
    X, Y = np.meshgrid(x, x)
    Z = multivariate_normal([0, 0], sample["wind_covar"]).pdf(np.dstack((X, Y)))
    axes[2].contourf(x, x, Z / Z.max(), levels=20, cmap="BuPu")
    axes[2].set_aspect("equal")
    axes[2].set_title("wind covariance", fontsize=10, color="dimgray")

    return axes


def animate_scene(sample, resolution=128):
    """interactive plotly figure of a scene with a slider over the timesteps (the "oil site overview")"""
    x, y, Z = concentration_field(sample, resolution)
    Z = Z.round(decimals=2)

    n_sensors, n_leaks, n_sources = int(sample["n_sensors"]), int(sample["n_leaks"]), int(sample["n_sources"])
    pos_sources, pos_leaks, pos_sensors = sample["pos_sources"][:n_sources], sample["pos_leaks"][:n_leaks], sample["pos_sensors"][:n_sensors]
    wind_speed, wind_direc = sample["wind_speed"], sample["wind_direc"]

    def traces(t):
        z_max = max(Z[t].max(), 1e-6)
        return [
            go.Contour(x=x, y=y, z=Z[t], colorscale="deep", line_width=0, contours=dict(start=0, end=z_max, size=z_max / 20), colorbar=dict(x=1.05, y=0.40, len=0.65)),
            go.Scatter(x=pos_sources[:, 0], y=pos_sources[:, 1], mode="markers", name=f"Sources ({n_sources})", marker=dict(color="black", size=6, symbol="circle", line=dict(width=1, color="lightgray"))),
            go.Scatter(x=pos_leaks[:, 0], y=pos_leaks[:, 1], mode="markers", name=f"Leaks ({n_leaks})", marker=dict(color="SeaGreen", size=12, symbol="star-triangle-up", line=dict(width=1, color="lightgray"))),
            go.Scatter(x=pos_sensors[:, 0], y=pos_sensors[:, 1], mode="markers", name=f"Sensors ({n_sensors})", marker=dict(color="firebrick", size=12, symbol="pentagon", line=dict(width=1, color="lightgray"))),
            go.Scatter(x=[None], y=[None], name=f"W. Speed: {wind_speed[t]:.2f}", mode="markers", marker_symbol="triangle-right", marker_size=8, marker_color="MediumOrchid"),
        ]  # fmt: skip

    def annotations(t):
        arrow = dict(ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2)
        result = [
            # wind arrow (gray outline + colored arrow)
            dict(x=(wind_speed[t] + 0.23) * 0.8 * np.cos(wind_direc[t]), y=(wind_speed[t] + 0.23) * 0.8 * np.sin(wind_direc[t]), arrowsize=0.8, arrowwidth=5, arrowcolor="lightgray", **arrow),
            dict(x=(wind_speed[t] + 0.2) * 0.8 * np.cos(wind_direc[t]), y=(wind_speed[t] + 0.2) * 0.8 * np.sin(wind_direc[t]), arrowsize=1.0, arrowwidth=3, arrowcolor="MediumOrchid", **arrow),
        ]  # fmt: skip
        label = dict(xref="x", yref="y", showarrow=False, xanchor="center", yanchor="top")
        # leak rates and the (normalized) concentration that each sensor reports
        for idx, (px, py) in enumerate(pos_leaks):
            result.append(dict(x=px, y=py - 0.03, text=f"Q: {sample['Q_leaks'][t, idx]:.2f}", font=dict(size=10, family=FONT, color="SeaGreen"), **label))
        for idx, (px, py) in enumerate(pos_sensors):
            result.append(dict(x=px, y=py - 0.03, text=f"C: {sample['con_sensors'][t, idx]:.2f}", font=dict(size=10, family=FONT, color="firebrick"), **label))
        return result

    fig = go.Figure(data=traces(0))
    fig.frames = [go.Frame(data=traces(t), name=f"{int(t)}", layout=go.Layout(annotations=annotations(t))) for t in sample["time"]]

    fig.update_xaxes(range=[-1.2, 1.2], scaleanchor="y", scaleratio=1, constrain="domain")
    fig.update_yaxes(range=[-1.2, 1.2], scaleanchor="x", scaleratio=1, constrain="domain")
    fig.update_layout(
        width=600, height=600, annotations=annotations(0),
        title=dict(text="----- Oil Site Overview -----", x=0.5, y=0.9, xanchor="center", yanchor="top", font=dict(family=FONT, size=26, color="black")),
        legend=dict(x=1.05, y=0.93, xanchor="left", yanchor="top", font=dict(family=FONT, size=12, color="black")),
        sliders=[
            dict(
                active=0, transition=dict(duration=0), x=0.0, y=0.05, xanchor="left", yanchor="top", len=1.45, currentvalue=dict(prefix="Timestep: "),
                # the frame name in args has to match up EXACTLY with the name of a frame
                steps=[dict(method="animate", label=f"{int(t)}", args=[[f"{int(t)}"], dict(mode="immediate", frame=dict(duration=0, redraw=True))]) for t in sample["time"]],
            )
        ],
    )  # fmt: skip

    return fig

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
import scipy.special
import sympy as sp


def plot_pdf(ax, pdf_function, xlim=(-3.0, 3.0), ylim=(-3.0, 3.0), resolution=200, cmap="viridis"):
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)
    XY = np.stack([X, Y], axis=-1)

    # Evaluate pdf on grid
    Z = np.apply_along_axis(pdf_function, -1, XY)

    contour = ax.contourf(X, Y, Z, levels=30, cmap=cmap)
    #plt.colorbar(contour, ax=ax)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal', adjustable='box')


def build_symbolic_fields(f_expr_builder, dimension: int):
    # Symbols for x and u
    x_symbols = sp.symbols('x0:%d' % dimension)
    u_symbols = sp.symbols('u0:%d' % dimension)

    # Build vector field f(x) as sympy expressions
    f_exprs = f_expr_builder(x_symbols)
    f_vec = sp.Matrix(f_exprs)

    # div_x f(x)
    div_x_expr = sum(sp.diff(f_vec[i], x_symbols[i]) for i in range(dimension))

    # Standard normal helpers
    def phi_expr(x):
        return (1.0 / sp.sqrt(2.0 * sp.pi)) * sp.exp(-x**2 / 2.0)

    def invPhi_expr(u):
        return sp.sqrt(2.0) * sp.erfinv(2.0*u - 1.0)

    # Build u-dependent expressions via x(u) (component-wise quantile)
    x_of_u = [invPhi_expr(u_symbols[i]) for i in range(dimension)]

    # u̇_i(u) = φ(x_i(u)) * f_i(x(u))
    u_vec_exprs = [phi_expr(x_symbols[i]) * f_vec[i] for i in range(dimension)]
    u_vec_exprs = [expr.subs({x_symbols[j]: x_of_u[j] for j in range(dimension)}) for expr in u_vec_exprs]

    # div_u(u̇) = div_x f(x) - x·f(x) evaluated at x(u)
    x_dot_f_expr = sum(x_symbols[i] * f_vec[i] for i in range(dimension))
    div_u_expr = (div_x_expr - x_dot_f_expr).subs({x_symbols[j]: x_of_u[j] for j in range(dimension)})

    # Numeric callables (wrappers to accept np.ndarray)
    modules = [{
        'erf': scipy.special.erf,
        'erfinv': scipy.special.erfinv,
        'sqrt': np.sqrt,
        'exp': np.exp,
        'pi': np.pi
    }, 'numpy']

    f_lam = sp.lambdify(x_symbols, list(f_vec), modules=modules)
    div_x_lam = sp.lambdify(x_symbols, div_x_expr, modules=modules)
    u_vec_lam = sp.lambdify(u_symbols, u_vec_exprs, modules=modules)
    div_u_lam = sp.lambdify(u_symbols, div_u_expr, modules=modules)

    def vector_field(x: np.ndarray):
        return np.array(f_lam(*[x[i] for i in range(dimension)]), dtype=float)

    def divergence(x: np.ndarray):
        return float(div_x_lam(*[x[i] for i in range(dimension)]))

    def u_vector_field(u: np.ndarray):
        return np.array(u_vec_lam(*[u[i] for i in range(dimension)]), dtype=float)

    def u_divergence(u: np.ndarray):
        return float(div_u_lam(*[u[i] for i in range(dimension)]))

    return vector_field, divergence, u_vector_field, u_divergence

class Region:
    def __init__(self, mins : np.ndarray, maxes : np.ndarray):
        self.points = []

        self.mins = mins
        self.maxes = maxes

        # Generate boundary points for a 2D axis-aligned rectangle
        # defined by mins = [x_min, y_min], maxes = [x_max, y_max]
        if mins.shape[0] == 2 and maxes.shape[0] == 2:
            x_min, y_min = mins[0], mins[1]
            x_max, y_max = maxes[0], maxes[1]

            num_per_edge = 200

            xs = np.linspace(x_min, x_max, num_per_edge)
            ys = np.linspace(y_min, y_max, num_per_edge)

            # Bottom edge (y = y_min)
            bottom = np.stack([xs, np.full_like(xs, y_min)], axis=1)
            # Top edge (y = y_max)
            top = np.stack([xs, np.full_like(xs, y_max)], axis=1)
            # Left edge (x = x_min) excluding corners to avoid duplicates
            left = np.stack([np.full_like(ys, x_min), ys], axis=1)[1:-1]
            # Right edge (x = x_max) excluding corners
            right = np.stack([np.full_like(ys, x_max), ys], axis=1)[1:-1]

            self.points = np.concatenate([bottom, right, top[::-1], left[::-1]], axis=0)
        else:
            # For non-2D cases, leave empty for now
            self.points = np.empty((0, mins.shape[0]))
    
    def contains(self, x : np.ndarray):
        return np.all(x >= self.mins) and np.all(x <= self.maxes)
    
    def volume(self):
        return np.prod(self.maxes - self.mins)

    def plot(self, ax : plt.Axes, color: str = 'k', linewidth: float = 1.5, alpha: float = 0.7):
        # Points are stored in counter-clockwise order around the rectangle
        if self.points is None or self.points.size == 0:
            return
        poly = MplPolygon(self.points, closed=True, edgecolor=color, facecolor=color, linewidth=linewidth, alpha=alpha)
        ax.add_patch(poly)
    
    def flow_backward(self, vector_field : callable, dT : float):
        for i in range(self.points.shape[0]):
            self.points[i, :] -= vector_field(self.points[i, :]) * dT

if __name__ == "__main__":
    cmap = "Blues"
    region_color = "deeppink"
    #def vector_field(x : np.ndarray):
    #    return np.array([x[1], (1.0 - x[0]**2) * x[1] - x[0]])

    #def divergence(x : np.ndarray):
    #    return (1.0 - x[0]**2)
    
    # Build vector field and derived quantities from SymPy
    def f_expr_builder(xs):
        x0, x1 = xs
        return (
            1.0 - x1**2,
            sp.sin(2.0*(x1 - x0))
            #sp.sin(6.0*x0)
            #0.2*x0 + 0.8*x1**2 - 0.3 + sp.sin(2.0*x0 + 0.2) - 0.5
        )

    vector_field, divergence, u_vector_field, u_divergence = build_symbolic_fields(f_expr_builder=f_expr_builder, dimension=2)

    target_region_bounds = [np.array([0.1, -0.7]), np.array([0.6, -0.2])]

    def ss():
        n_timesteps = 5
        dT = 0.15
        n_sub_steps = 30
        figs_ss = []

        # Define a 2D Gaussian PDF and plot it on ax_0
        mu = np.array([0.0, 0.0])
        Sigma = 0.1 * np.eye(2)
        inv_Sigma = np.linalg.inv(Sigma)
        det_Sigma = np.linalg.det(Sigma)
        norm_const = 1.0 / (2.0 * np.pi * np.sqrt(det_Sigma))
        def init_pdf(x: np.ndarray):
            d = x - mu
            return norm_const * np.exp(-0.5 * d @ inv_Sigma @ d)

        for k in range(n_timesteps - 1, -1, -1):
            fig_k, ax_k = plt.subplots(figsize=(6, 6))
            ax_k.set_xticks([])
            ax_k.set_yticks([])

            # Create a fresh region for this timestep
            target_region = Region(mins=target_region_bounds[0], maxes=target_region_bounds[1])
            # Flow backward to the correct time
            for _ in range(n_timesteps - 1 - k):
                target_region.flow_backward(vector_field, dT)

            def pdf_k(x : np.ndarray):
                log_density = 0.0
                for i in range(k * n_sub_steps):
                    log_density -= divergence(x) * dT / n_sub_steps
                    x -= vector_field(x) * dT / n_sub_steps
                log_density += np.log(init_pdf(x))
                if np.isnan(log_density) or np.isinf(log_density):
                    return 0.0
                return np.exp(log_density)


            plot_pdf(ax_k, pdf_k, xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), resolution=50, cmap=cmap)
            target_region.plot(ax_k, color=region_color, linewidth=1.0)
            
            plt.tight_layout()
            figs_ss.append(fig_k)

        return figs_ss

    def us():
        gaussian_cdf = lambda x : 0.5 * (1.0 + scipy.special.erf(x / np.sqrt(2.0)))
        gaussian_quantile = lambda x : np.sqrt(2.0) * scipy.special.erfinv(2.0 * x - 1.0)
        gaussian_pdf = lambda x : 1.0 / np.sqrt(2.0 * np.pi) * np.exp(-0.5 * x**2)

        #def u_vector_field(u : np.ndarray):
        #    x = gaussian_quantile(u)
        #    return gaussian_pdf(x) * vector_field(x)
        #def u_divergence(u : np.ndarray):
        #    x = gaussian_quantile(u)

        # Use sympy-derived u-space dynamics (already built above)



        n_timesteps = 5
        dT = 0.15
        n_sub_steps = 30
        figs_us = []

        # Define a 2D Gaussian PDF and plot it on ax_0
        mu = np.array([0.0, 0.0])
        Sigma = 0.1 * np.eye(2)
        inv_Sigma = np.linalg.inv(Sigma)
        det_Sigma = np.linalg.det(Sigma)
        norm_const = 1.0 / (2.0 * np.pi * np.sqrt(det_Sigma))
        def init_pdf(x: np.ndarray):
            return 1.0

        for k in range(n_timesteps - 1, -1, -1):
            fig_k, ax_k = plt.subplots(figsize=(6, 6))
            ax_k.set_xticks([])
            ax_k.set_yticks([])

            # Create a fresh region for this timestep
            target_region = Region(mins=gaussian_cdf(target_region_bounds[0]), maxes=gaussian_cdf(target_region_bounds[1]))
            # Flow backward to the correct time
            for _ in range(n_timesteps - 1 - k):
                target_region.flow_backward(u_vector_field, dT)

            def pdf_k(x : np.ndarray):
                log_density = 0.0
                for i in range(k * n_sub_steps):
                    log_density -= u_divergence(x) * dT / n_sub_steps
                    x -= u_vector_field(x) * dT / n_sub_steps
                log_density += np.log(init_pdf(x))
                if np.isnan(log_density) or np.isinf(log_density):
                    return 0.0
                return np.exp(log_density)


            plot_pdf(ax_k, pdf_k, xlim=(0.0, 1.0), ylim=(0.0, 1.0), resolution=50, cmap=cmap)
            target_region.plot(ax_k, color=region_color, linewidth=1.0)
            
            plt.tight_layout()
            figs_us.append(fig_k)

        return figs_us

    figs_ss = ss()
    figs_us = us()

    for i, fig in enumerate(figs_ss):
        fig.savefig(f"../figures/ss_{i}.pdf")
    for i, fig in enumerate(figs_us):
        fig.savefig(f"../figures/us_{i}.pdf")

    plt.show()
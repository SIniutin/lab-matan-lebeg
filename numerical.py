import math
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# f(x) = x^2, E = [1, 4], F(x) = sqrt(x)
f = lambda x: x ** 2
F = lambda x: math.sqrt(x)

EXACT_LEBESGUE  = 21.0
EXACT_STIELTJES = 31.0 / 5.0


def lebesgue_integral(n):
    """Интеграл Лебега от f_n по E: сумма Римана левыми концами."""
    total = 0.0
    dx = 3.0 / n
    for k in range(n):
        xk = 1.0 + 3.0 * k / n
        total += f(xk) * dx
    return total


def stieltjes_integral(n):
    """Интеграл Лебега–Стилтьеса от f_n по E относительно F(x)=sqrt(x)."""
    total = 0.0
    for k in range(n):
        xk  = 1.0 + 3.0 * k / n
        xk1 = 1.0 + 3.0 * (k + 1) / n
        total += f(xk) * (F(xk1) - F(xk))
    return total


def print_table(title, ns, func, exact):
    """Вывод таблицы значений и погрешностей."""
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print(f"{'=' * 55}")
    print(f"  {'n':>8}  {'Значение':>14}  {'Погрешность':>14}")
    print(f"  {'-' * 50}")
    for n in ns:
        val = func(n)
        err = abs(val - exact)
        print(f"  {n:>8}  {val:>14.8f}  {err:>14.2e}")
    print(f"  {'∞ (аналит.)':>8}  {exact:>14.8f}  {'0':>14}")
    print(f"{'=' * 55}")


def plot_fn(ns_plot, save_path="graphs.png"):
    """Строим графики простых функций f_n рядом с f(x)=x^2."""
    x_dense = np.linspace(1, 4, 500)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for idx, n in enumerate(ns_plot):
        ax = axes[idx]
        dx = 3.0 / n

        for k in range(n):
            xk  = 1.0 + 3.0 * k / n
            fk  = f(xk)
            ax.add_patch(mpatches.Rectangle(
                (xk, 0), dx, fk,
                linewidth=0.5,
                edgecolor='steelblue',
                facecolor='lightsteelblue',
                alpha=0.7
            ))

        # f(x) = x^2
        ax.plot(x_dense, x_dense ** 2, 'r-', linewidth=1.5, label=r'$f(x)=x^2$')
        ax.set_title(f'$f_{{n}}$, $n={n}$')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(1, 4)
        ax.set_ylim(0, 18)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(r'Простые функции $f_n(x)$ при различных $n$', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nГрафик сохранён: {save_path}")


if __name__ == "__main__":
    t0 = time.time()

    print_table(
        "Интеграл Лебега",
        [10, 100, 1000],
        lebesgue_integral,
        EXACT_LEBESGUE,
    )

    print_table(
        "Интеграл Лебега–Стилтьеса",
        [50, 500, 5000],
        stieltjes_integral,
        EXACT_STIELTJES,
    )

    elapsed = time.time() - t0
    print(f"\nВремя работы: {elapsed:.4f} сек")

    plot_fn([1, 5, 20, 50])
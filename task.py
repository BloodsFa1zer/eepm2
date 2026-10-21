import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Задача про вилов риби у ставку, варіант 15
# dx/dt = r*x*(1 - x/q) - p,  p > 0 - квота на вилов
N = 15
r = 3 * N        # N в [9;16]
q = 300 * N
k = r / q
p_crit = r * q / 4
print("r =", r, " q =", q, " rq/4 =", p_crit, " k = r/q =", k)

def f(x, p):
    return r * x * (1 - x / q) - p

# рівняння переписуємо як dx/dt = -k*(x^2 - q*x + p/k),
# стаціонарні точки x1,2 = (q -+ sqrt(D))/2, де D = q^2 - 4*p/k
a2 = 10000

# випадок 1: p = rq/4, D = 0, одна (парна) точка x* = q/2
p1 = p_crit
x_star = q / 2
print("\nВипадок 1: p =", p1, " D =", q**2 - 4 * p1 / k, " x* =", x_star)

# випадок 2: p = rq/4 - a^2, D > 0, дві точки
p2 = p_crit - a2
D2 = q**2 - 4 * p2 / k
eps2 = np.sqrt(D2) / 2
x1, x2 = q / 2 - eps2, q / 2 + eps2
print("Випадок 2: p =", p2, " D =", D2, " x1 =", x1, " x2 =", x2)
print("  f'(x1) =", r * (1 - 2 * x1 / q), "> 0 - нестійка;  f'(x2) =", r * (1 - 2 * x2 / q), "< 0 - стійка")

# випадок 3: p = rq/4 + a^2, D < 0, дійсних стаціонарних точок немає
p3 = p_crit + a2
D3 = q**2 - 4 * p3 / k
eps3 = np.sqrt(-D3) / 2
print("Випадок 3: p =", p3, " D =", D3, " (точок немає)")

# точні розв'язки (заміна u = x - q/2)
def sol1(t, x0):             # du/dt = -k*u^2
    u0 = x0 - q / 2
    return q / 2 + u0 / (1 + k * u0 * t)

def sol2(t, x0):             # du/dt = -k*(u^2 - eps^2)
    u0 = x0 - q / 2
    z = u0 / eps2
    if abs(z) < 1:
        return q / 2 + eps2 * np.tanh(k * eps2 * t + np.arctanh(z))
    c = 0.5 * np.log((z + 1) / (z - 1))      # arccoth(z)
    return q / 2 + eps2 / np.tanh(k * eps2 * t + c)

def sol3(t, x0):             # du/dt = -k*(u^2 + eps^2)
    u0 = x0 - q / 2
    return q / 2 + eps3 * np.tan(np.arctan(u0 / eps3) - k * eps3 * t)

# чисельний розв'язок; зупиняємось, коли x = 0 (популяція зникла)
def num(p, x0, T):
    ev = lambda t, x: x[0]
    ev.terminal = True
    ev.direction = -1
    return solve_ivp(lambda t, x: [f(x[0], p)], [0, T], [x0],
                     rtol=1e-11, atol=1e-9, events=ev, dense_output=True)

cases = [
    (1, p1, sol1, [1500, 3500], 1.0),
    (2, p2, sol2, [800, 2000, 4000], 0.5),
    (3, p3, sol3, [q / 3, 3 * q / 4], 0.25),
]

for n, p, sol, x0s, T in cases:
    print("\n випадок", n)
    for x0 in x0s:
        s = num(p, x0, T)
        tz = s.t_events[0][0] if len(s.t_events[0]) else None
        t_ = T / 2 if tz is None else tz / 2
        print(f"x0={x0:.0f}: t_зникнення={'немає' if tz is None else round(tz, 4)}, "
              f"x(t={t_:.4f}) формула={sol(t_, x0):.4f} чисельно={s.sol(t_)[0]:.4f}, x(T)={s.y[0][-1]:.2f}")

# ---------- графіки ----------
def plot_case(p, x0s, T, eq, fname, title):
    plt.figure(figsize=(7, 4.2))
    cols = ['tab:red', 'tab:green', 'tab:blue']
    for x0, c in zip(x0s, cols):
        s = num(p, x0, T)
        tt = np.linspace(0, s.t[-1], 800)
        plt.plot(tt, s.sol(tt)[0], c, label=f'x(0) = {x0:.0f}')
        plt.plot(0, x0, 'o', color=c)
        if len(s.t_events[0]):
            plt.plot(s.t_events[0][0], 0, 'x', color=c, markersize=9)
    for e in eq:
        plt.axhline(e, color='gray', linestyle='--', linewidth=1)
        plt.text(T * 1.005, e, f'{e:.0f}', va='center', fontsize=8)
    plt.axhline(q, color='k', linestyle=':', linewidth=1)
    plt.text(T * 1.005, q, f'q={q}', va='center', fontsize=8)
    plt.xlim(0, T)
    plt.ylim(-100, q + 250)
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.title(title)
    plt.grid(alpha=.4)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

plot_case(p1, [1500, 3500], 1.0, [x_star], 'c1.png', f'Випадок 1: p = {p1:.0f}')
plot_case(p2, [800, 2000, 4000], 0.5, [x1, x2], 'c2.png', f'Випадок 2: p = {p2:.0f}')
plot_case(p3, [q / 3, 3 * q / 4], 0.25, [], 'c3.png', f'Випадок 3: p = {p3:.0f}')

# права частина рівняння для трьох квот
xs = np.linspace(0, q, 400)
plt.figure(figsize=(7, 4))
for p, lab in ((p1, 'p = rq/4'), (p2, 'p = rq/4 - a²'), (p3, 'p = rq/4 + a²')):
    plt.plot(xs, f(xs, p), label=f'{lab} = {p:.0f}')
plt.axhline(0, color='k', linewidth=.8)
plt.xlabel('x')
plt.ylabel('dx/dt')
plt.title('Права частина рівняння f(x)')
plt.grid(alpha=.4)
plt.legend()
plt.tight_layout()
plt.savefig('fx.png', dpi=150)
plt.close()

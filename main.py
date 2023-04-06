import math
import scipy.integrate
import matplotlib.pyplot as plt
import numpy as np
import warnings
import matplotlib.cbook


def f(x, n):
    return x ** n * math.exp(-x ** 2 / n)


def fourier_coeffs(n, k_max):
    a_coeffs = []
    b_coeffs = []
    for k in range(k_max + 1):
        integrand = lambda x: f(x, n) * np.cos(k * x)
        integral, _ = scipy.integrate.quad(integrand, -math.pi, math.pi)
        a_k = 1 / math.pi * integral
        a_coeffs.append(a_k)
        if k > 0:
            integrand = lambda x: f(x, n) * np.sin(k * x)
            integral, _ = scipy.integrate.quad(integrand, -math.pi, math.pi)
            b_k = 1 / math.pi * integral
            b_coeffs.append(b_k)
    return a_coeffs, b_coeffs


def print_table(a_coeffs, b_coeffs):
    print(f"{'k':>2} {'a_k':>15} {'b_k':>15}")
    print("-" * 45)
    print(f"{0:2d} {a_coeffs[0]:15.6f}")
    for k in range(1, len(a_coeffs)):
        print(f"{k:2d} {a_coeffs[k]:15.6f} {b_coeffs[k - 1]:15.6f}")
    print()


n = 10
k_max = 24

print("\nЗначення ряду Фур'є, при n=24: ", f(n, k_max), "\n")
a_coeffs, b_coeffs = fourier_coeffs(n, k_max)
print_table(a_coeffs, b_coeffs)

warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

# обчислення коефіцієнтів Фур'є

a_coeffs, b_coeffs = fourier_coeffs(n, k_max)


# функція для обчислення наближення рядом Фур'є
def fourier_series(x, n, k_max, a_coeffs, b_coeffs):
    series_sum = a_coeffs[0] / 2
    for k in range(1, k_max + 1):
        series_sum += a_coeffs[k] * np.cos(k * x) + b_coeffs[k - 1] * np.sin(k * x)
    return series_sum


# побудова графіку
x_vals = np.linspace(-np.pi, np.pi , 500)
y_vals = [f(x, n) for x in x_vals]
y_series_vals = [fourier_series(x, n, k_max, a_coeffs, b_coeffs) for x in x_vals]


def relative_error(x_vals, y_vals, y_series_vals):
    f_norm = np.linalg.norm(y_vals)
    error_norm = np.linalg.norm(np.array(y_vals) - np.array(y_series_vals))
    return error_norm / f_norm


def save_results_to_files(a_coeffs, b_coeffs, n, k_max, rel_error):
    if len(a_coeffs) > len(b_coeffs):
        b_coeffs.extend([0.0] * (len(a_coeffs) - len(b_coeffs)))
    data = np.column_stack((a_coeffs, b_coeffs))
    header = f"N = {n}, k_max = {k_max}"
    np.savetxt("results.txt", data, header=header, fmt="%.8f", delimiter="\t")
    with open("relative_error.txt", "w") as f:
        f.write(f"Relative error:\t{rel_error:.8f}")


def fourier_series_freq_domain(k, n, k_max, a_coeffs, b_coeffs):
    series_sum = a_coeffs[0] / 2
    for i in range(1, k_max + 1):
        series_sum += a_coeffs[i] * np.cos(i * k) + b_coeffs[i - 1] * np.sin(i * k)
    return series_sum


x_vals = np.linspace(-np.pi, np.pi, 500)
rel_error = relative_error(x_vals, y_vals, y_series_vals)
print(f"Relative error: {rel_error:.5f}")

k_vals = np.arange(k_max + 1)
a_vals = np.array(a_coeffs)
b_vals = np.concatenate(([0], b_coeffs))

plt.figure(figsize=(10, 7))

plt.subplot(2, 1, 1)
plt.stem(k_vals, a_vals, use_line_collection=True)
plt.title("Cosine Coefficients (a_k)")
plt.xlabel("k")
plt.ylabel("a_k")

plt.subplot(2, 1, 2)
plt.stem(k_vals, b_vals, use_line_collection=True)
plt.title("Sine Coefficients (b_k)")
plt.xlabel("k")
plt.ylabel("b_k")

plt.tight_layout()
plt.show()

k_vals_freq = np.linspace(0, k_max, 500)
y_vals_freq = [fourier_series_freq_domain(k, n, k_max, a_coeffs, b_coeffs) for k in k_vals_freq]

plt.figure(figsize=(7, 5))

plt.plot(k_vals_freq, y_vals_freq, label=f'Fourier series, N={k_max}', color="red")
plt.xlabel('k')
plt.ylabel('y')
plt.legend()
plt.show()

save_results_to_files(a_coeffs, b_coeffs, n, k_max, rel_error)

# побудова графіку


plt.plot(x_vals, y_vals, label='f(x)', color="red")
plt.plot(x_vals, y_series_vals, label=f'Fourier series, N={k_max}', color="green")

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


def fourier_series(x, n, k_max, a_coeffs, b_coeffs):
    series_sum = np.zeros_like(x)
    for k in range(1, k_max + 1):
        series_sum += a_coeffs[k] * np.cos(k * x) + b_coeffs[k - 1] * np.sin(k * x)
        yield series_sum


y_vals = [f(x, n) for x in x_vals]
y_series_vals = fourier_series(x_vals, n, 10, a_coeffs, b_coeffs)
plt.plot(x_vals, y_vals, label='f(x)')
for k, y_k in enumerate(y_series_vals):
    plt.plot(x_vals, y_k, label=f'N_{k + 1}(x)')
plt.legend()
plt.show()

# Обчислюємо значення функції для кожного x
f_vals = x_vals ** n * np.exp(-x_vals ** 2 / n)

# Обчислюємо значення ряду Фур'є для функції
N = 50
a_vals = np.zeros(N)
b_vals = np.zeros(N)
for n in range(1, N + 1):
    a_vals[n - 1] = 2 * np.sum(f_vals * np.cos(n * x_vals)) / len(x_vals)
    b_vals[n - 1] = 2 * np.sum(f_vals * np.sin(n * x_vals)) / len(x_vals)
y_series_vals = np.zeros(len(x_vals))
for n in range(1, N + 1):
    y_series_vals += a_vals[n - 1] * np.cos(n * x_vals) + b_vals[n - 1] * np.sin(n * x_vals)

# Обчислюємо значення відносної помилки для кожного x
errors = []
for i in range(len(x_vals)):
    errors.append(relative_error(x_vals[i], f_vals[i], y_series_vals[i]))

# Побудова графіка відносної помилки
plt.plot(x_vals, errors)
plt.xlabel('x')
plt.ylabel('Relative error')
plt.show()

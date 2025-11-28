#include <vector>
#include <cmath>
#include <stdexcept>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

const double EPS = 1e-12;

std::vector<double> gauss_jordan(std::vector<std::vector<double>> a,
                                 std::vector<double> b)
{
    int n = a.size();
    if ((int)b.size() != n)
        throw std::runtime_error("Размеры A и b не совпадают");

    for (int i = 0; i < n; i++)
        a[i].push_back(b[i]);

    for (int col = 0; col < n; col++) {

        int sel = col;
        for (int row = col; row < n; row++)
            if (fabs(a[row][col]) > fabs(a[sel][col]))
                sel = row;

        if (fabs(a[sel][col]) < EPS)
            throw std::runtime_error("Система не имеет уникального решения");

        std::swap(a[col], a[sel]);

        double piv = a[col][col];
        for (int j = col; j <= n; j++)
            a[col][j] /= piv;

        for (int row = 0; row < n; row++) {
            if (row == col) continue;
            double factor = a[row][col];
            for (int j = col; j <= n; j++)
                a[row][j] -= factor * a[col][j];
        }
    }

    std::vector<double> x(n);
    for (int i = 0; i < n; i++)
        x[i] = a[i][n];
    return x;
}


py::array_t<double> solve_numpy(py::array_t<double> A_in,
                                py::array_t<double> b_in)
{
    if (A_in.ndim() != 2)
        throw std::runtime_error("A не матрица");
    if (b_in.ndim() != 1)
        throw std::runtime_error("b не вектор");

    size_t n = A_in.shape(0);
    if (A_in.shape(1) != n)
        throw std::runtime_error("A не квадратная");
    if (b_in.shape(0) != n)
        throw std::runtime_error("Размеры A и b отличаются");

    std::vector<std::vector<double>> A(n, std::vector<double>(n));
    std::vector<double> b(n);

    auto A_buf = A_in.unchecked<2>();
    auto b_buf = b_in.unchecked<1>();

    for (size_t i = 0; i < n; i++) {
        b[i] = b_buf(i);
        for (size_t j = 0; j < n; j++)
            A[i][j] = A_buf(i, j);
    }

    auto result = gauss_jordan(A, b);

    py::array_t<double> out(n);
    auto out_buf = out.mutable_unchecked<1>();
    for (size_t i = 0; i < n; i++)
        out_buf(i) = result[i];

    return out;
}

PYBIND11_MODULE(gauss, m) {
    m.doc() = "Гаусс-Жордан";

    m.def("solve", &solve_numpy,
          "Решение Ax=b методом Гаусса–Жордана");
}

import numpy as np
import gauss
import pytest
import time

def test_gauss_solve_simple():
    A = np.array([[2.0, 1.0, -1.0],
                  [-3.0, -1.0, 2.0],
                  [-2.0, 1.0, 2.0]])
    b = np.array([8.0, -11.0, -3.0])
    
    x_expected = np.linalg.solve(A, b)
    x_actual = gauss.solve(A, b)
    
    np.testing.assert_allclose(x_actual, x_expected, rtol=1e-5, atol=1e-8)

def test_gauss_solve_larger_matrix():
    np.random.seed(42)
    A = np.random.rand(5, 5) * 10
    b = np.random.rand(5) * 10
    
    x_expected = np.linalg.solve(A, b)
    x_actual = gauss.solve(A, b)
    
    np.testing.assert_allclose(x_actual, x_expected, rtol=1e-5, atol=1e-8)

def test_gauss_solve_identity_matrix():
    A = np.eye(3)
    b = np.array([1.0, 2.0, 3.0])
    
    x_expected = np.linalg.solve(A, b)
    x_actual = gauss.solve(A, b)
    
    np.testing.assert_allclose(x_actual, x_expected, rtol=1e-5, atol=1e-8)

def test_gauss_solve_diagonal_matrix():
    A = np.diag([2.0, 3.0, 4.0])
    b = np.array([4.0, 9.0, 16.0])
    
    x_expected = np.linalg.solve(A, b)
    x_actual = gauss.solve(A, b)
    
    np.testing.assert_allclose(x_actual, x_expected, rtol=1e-5, atol=1e-8)

def test_gauss_solve_singular_matrix():
    A = np.array([[1.0, 1.0],
                  [1.0, 1.0]])
    b = np.array([1.0, 2.0])
    
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.solve(A, b)
    
    with pytest.raises(RuntimeError, match="Система не имеет уникального решения"):
        gauss.solve(A, b)

def test_gauss_solve_inconsistent_system():
    A = np.array([[1.0, 1.0],
                  [2.0, 2.0]])
    b = np.array([1.0, 3.0])
    
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.solve(A, b)
    
    with pytest.raises(RuntimeError, match="Система не имеет уникального решения"):
        gauss.solve(A, b)

def test_gauss_solve_large_random_matrix():
    np.random.seed(123)
    size = 10
    A = np.random.rand(size, size) * 100
    b = np.random.rand(size) * 100
    
    x_expected = np.linalg.solve(A, b)
    x_actual = gauss.solve(A, b)
    
    np.testing.assert_allclose(x_actual, x_expected, rtol=1e-5, atol=1e-8)

def test_performance_comparison():
    print("\n--- Performance Comparison ---")
    np.random.seed(42)
    size = 200
    A = np.random.rand(size, size) * 100
    b = np.random.rand(size) * 100
    gauss.solve(A, b)
    np.linalg.solve(A, b)

    start_time = time.perf_counter()
    x_gauss = gauss.solve(A, b)
    end_time = time.perf_counter()
    gauss_time = end_time - start_time
    print(f"gauss.solve for {size}x{size} matrix: {gauss_time:.6f} seconds")

    start_time = time.perf_counter()
    x_numpy = np.linalg.solve(A, b)
    end_time = time.perf_counter()
    numpy_time = end_time - start_time
    print(f"numpy.linalg.solve for {size}x{size} matrix: {numpy_time:.6f} seconds")

    np.testing.assert_allclose(x_gauss, x_numpy, rtol=1e-5, atol=1e-8)
#include <vector>
#include <iostream>
#include <stdexcept> // For std::out_of_range

class Matrix {
public:
    int rows;
    int cols;
    std::vector<double> data;

    // Constructor
    Matrix(int r, int c) : rows(r), cols(c), data(r * c) {
        if (r <= 0 || c <= 0) {
            throw std::invalid_argument("Matrix dimensions must be positive.");
        }
    }

    // Get element at (row, col)
    double get(int r, int c) const {
        if (r < 0 || r >= rows || c < 0 || c >= cols) {
            throw std::out_of_range("Matrix element access out of bounds.");
        }
        return data[r * cols + c];
    }

    // Set element at (row, col)
    void set(int r, int c, double value) {
        if (r < 0 || r >= rows || c < 0 || c >= cols) {
            throw std::out_of_range("Matrix element access out of bounds.");
        }
        data[r * cols + c] = value;
    }

    // Fill matrix with a specific value
    void fill(double value) {
        std::fill(data.begin(), data.end(), value);
    }

    // Matrix multiplication
    Matrix multiply(const Matrix& other) const {
        if (cols != other.rows) {
            throw std::invalid_argument("Matrix dimensions are incompatible for multiplication.");
        }

        Matrix result(rows, other.cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < other.cols; ++j) {
                double sum = 0.0;
                for (int k = 0; k < cols; ++k) {
                    sum += get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        return result;
    }
    
    // Element-wise addition
    Matrix add(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrices must have the same dimensions for addition.");
        }
        Matrix result(rows, cols);
        for (int i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] + other.data[i];
        }
        return result;
    }

    // Element-wise subtraction
    Matrix subtract(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrices must have the same dimensions for subtraction.");
        }
        Matrix result(rows, cols);
        for (int i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] - other.data[i];
        }
        return result;
    }

    // Element-wise multiplication (Hadamard product)
    Matrix elementWiseMultiply(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrices must have the same dimensions for element-wise multiplication.");
        }
        Matrix result(rows, cols);
        for (int i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] * other.data[i];
        }
        return result;
    }

    // Scalar multiplication
    Matrix scalarMultiply(double scalar) const {
        Matrix result(rows, cols);
        for (int i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] * scalar;
        }
        return result;
    }

    // Transpose
    Matrix transpose() const {
        Matrix result(cols, rows);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result.set(j, i, get(i, j));
            }
        }
        return result;
    }

    // Print matrix (for debugging)
    void print() const {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                std::cout << get(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }
};

// Example Usage (can be put in a main function for testing)
/*
int main() {
    // Test Matrix creation and access
    Matrix m1(2, 3);
    m1.set(0, 0, 1.0);
    m1.set(0, 1, 2.0);
    m1.set(0, 2, 3.0);
    m1.set(1, 0, 4.0);
    m1.set(1, 1, 5.0);
    m1.set(1, 2, 6.0);
    std::cout << "Matrix m1:" << std::endl;
    m1.print();

    // Test Matrix multiplication
    Matrix m2(3, 2);
    m2.set(0, 0, 7.0);
    m2.set(0, 1, 8.0);
    m2.set(1, 0, 9.0);
    m2.set(1, 1, 1.0);
    m2.set(2, 0, 2.0);
    m2.set(2, 1, 3.0);
    std::cout << "\nMatrix m2:" << std::endl;
    m2.print();

    try {
        Matrix m3 = m1.multiply(m2);
        std::cout << "\nm1 * m2:" << std::endl;
        m3.print();
    } catch (const std::invalid_argument& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    // Test transpose
    Matrix m4(2, 2);
    m4.set(0,0,1); m4.set(0,1,2);
    m4.set(1,0,3); m4.set(1,1,4);
    std::cout << "\nMatrix m4:" << std::endl;
    m4.print();
    Matrix m4_t = m4.transpose();
    std::cout << "\nm4 Transposed:" << std::endl;
    m4_t.print();

    // Test addition
    Matrix ma1(2, 2);
    ma1.set(0,0,1); ma1.set(0,1,1);
    ma1.set(1,0,1); ma1.set(1,1,1);
    Matrix ma2(2, 2);
    ma2.set(0,0,2); ma2.set(0,1,2);
    ma2.set(1,0,2); ma2.set(1,1,2);
    Matrix ma_sum = ma1.add(ma2);
    std::cout << "\nMatrix addition (ma1 + ma2):" << std::endl;
    ma_sum.print();

    // Test scalar multiplication
    Matrix ms1(2, 2);
    ms1.set(0,0,1); ms1.set(0,1,2);
    ms1.set(1,0,3); ms1.set(1,1,4);
    Matrix ms_scaled = ms1.scalarMultiply(2.0);
    std::cout << "\nMatrix scalar multiply (ms1 * 2):" << std::endl;
    ms_scaled.print();

    // Test element-wise multiplication
    Matrix me1(2, 2);
    me1.set(0,0,1); me1.set(0,1,2);
    me1.set(1,0,3); me1.set(1,1,4);
    Matrix me2(2, 2);
    me2.set(0,0,5); me2.set(0,1,6);
    me2.set(1,0=7); me2.set(1,1,8);
    Matrix me_prod = me1.elementWiseMultiply(me2);
    std::cout << "\nMatrix element-wise multiplication (me1 * me2):" << std::endl;
    me_prod.print();

    return 0;
}
*/
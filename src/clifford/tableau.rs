use ndarray::{s, Array1, Array2};
use pyo3::prelude::*;

#[pyclass]
struct CliffordTableau {
    tableau: Array2<usize>,
    signs: Array1<usize>,
    n_qubits: usize,
}

fn mult_paulis(
    row_i: &Array1<usize>,
    row_j: &Array1<usize>,
    sign_i: usize,
    sign_j: usize,
    n_qubits: usize,
) -> (Array1<usize>, usize) {
    // Split the rows into x and z components
    let x_1 = row_i.slice(s![..n_qubits]).to_owned();
    let z_1 = row_i.slice(s![n_qubits..]).to_owned();
    let x_2 = row_j.slice(s![..n_qubits]).to_owned();
    let z_2 = row_j.slice(s![n_qubits..]).to_owned();

    // Perform element-wise multiplication and addition
    let x_1_z_2 = &z_1 * &x_2;
    let z_1_x_2 = &x_1 * &z_2;

    let ac = (&x_1_z_2 + &z_1_x_2) % 2;

    let x_1 = (&x_1 + &x_2) % 2;
    let z_1 = (&z_1 + &z_2) % 2;

    let x_1_z_2 = ((&x_1_z_2 + &x_1 + &z_1) % 2) * &ac;
    let sign_change = (((ac.sum() + 2 * x_1_z_2.sum()) % 4) > 1) as usize;
    let new_sign = (sign_i + sign_j + sign_change) % 4;

    let new_p1 = ndarray::concatenate![ndarray::Axis(0), x_1, z_1];

    (new_p1, new_sign)
}

impl CliffordTableau {
    fn _insert_pauli_row(&mut self, pauli: &Array1<usize>, p_sign: usize, row: usize) {
        let mut old_pauli = self.tableau.slice_mut(s![row, ..]);
        old_pauli.assign(pauli);
        self.signs[row] = p_sign;
    }
}


#[pymethods]
impl CliffordTableau {
    #[new]
    fn new(n_qubits: usize) -> Self {
        let tableau = Array2::<usize>::eye(2 * n_qubits);
        let signs = Array1::<usize>::zeros(2 * n_qubits);
        CliffordTableau { tableau, signs, n_qubits }
    }

    fn _x_out(&mut self, row: usize, col: usize) -> usize {
        // self.tableau[row, col] + 2 * self.tableau[row, col + self.n_qubits]
        self.tableau[(row, col)] + 2 * self.tableau[(row, col + self.n_qubits)]
    }

    fn _z_out(&mut self, row: usize, col: usize) -> usize {
        // self.tableau[row, col] + 2 * self.tableau[row, col + self.n_qubits]
        self.tableau[(row + self.n_qubits, col)] +
            2 * self.tableau[(row + self.n_qubits, col + self.n_qubits)]
    }

    fn _xor_row(&mut self, i: usize, j: usize) {
        let row_i = self.tableau.slice(s![i, ..]).to_owned();
        let row_j = self.tableau.slice(s![i, ..]).to_owned();

        let sign_i = self.signs[i];
        let sign_j = self.signs[j];

        let (new_p1, new_sign) = mult_paulis(&row_i, &row_j, sign_i, sign_j,
                                             self.n_qubits);

        self._insert_pauli_row(&new_p1, new_sign, i);
    }


    fn append_h(&mut self, qubit: usize) {
        let idx0 = qubit;
        let idx1 = self.n_qubits + qubit;

        self.signs = (&self.signs + &(&self.tableau.slice(s![.., idx0]) * &self.tableau.slice(s![.., idx1]))) % 2;
        let mut tableau_view = self.tableau.view_mut();
        for mut row in tableau_view.rows_mut() {
            row.swap(idx0, idx1);
        }
    }

    fn append_s(&mut self, qubit: usize) {
        let idx0 = qubit;
        let idx1 = self.n_qubits + qubit;

        // Update signs
        self.signs = (&self.signs + &(&self.tableau.slice(s![.., idx0]) * &self.tableau.slice(s![.., idx1]))) % 2;

        // XOR the columns idx0 and idx1
        let mut tableau_view = self.tableau.view_mut();
        for mut row in tableau_view.rows_mut() {
            row[idx1] ^= row[idx0];
        }
    }

    fn append_cnot(&mut self, control: usize, target: usize) {
        let x_ia = self.tableau.slice(s![.., control]).to_owned();
        let x_ib = self.tableau.slice(s![.., target]).to_owned();

        let z_ia = self.tableau.slice(s![.., self.n_qubits + control]).to_owned();
        let z_ib = self.tableau.slice(s![.., self.n_qubits + target]).to_owned();

        let control_n = control + self.n_qubits;
        let target_n = target + self.n_qubits;

        {
            let mut tableau_view = self.tableau.view_mut();
            for mut row in tableau_view.rows_mut() {
                row[target] ^= row[control];
                row[control_n] ^= row[target_n];
            }
        }

        self.signs = (&self.signs + &(x_ia * z_ib * (x_ib.clone() + z_ia.clone() + 1))) % 2;
    }

    fn prepend_h(&mut self, qubit: usize) {
        let idx0 = qubit;
        let idx1 = self.n_qubits + qubit;

        self.signs.swap(idx0, idx1);

        let mut tableau_view = self.tableau.view_mut();
        tableau_view.swap_axes(0, 1);
        for mut row in tableau_view.rows_mut() {
            row.swap(idx0, idx1);
        }
        tableau_view.swap_axes(0, 1);
    }

    fn prepend_s(&mut self, qubit: usize) {
        self._xor_row(qubit, qubit + self.n_qubits);
    }

    fn prepend_cnot(&mut self, control: usize, target: usize) {
        self._xor_row(control, target);
        self._xor_row(control + self.n_qubits, target + self.n_qubits);
    }

    // fn from_tableau(tableau: Array2<usize>, signs: Array1<usize>) -> CliffordTableau {
    //     let n_qubits = tableau.ncols() / 2;
    //     CliffordTableau { tableau, signs, n_qubits }
    // }
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use lazy_static::lazy_static;

    lazy_static! {
        static ref TABLEAU: Array2<usize> = array![
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ];
    }

    fn get_default_tableau() -> &'static Array2<usize> {
        &TABLEAU
    }

    #[test]
    fn test_mult_paulis() {
        let row_i = array![1, 0, 1, 0, 1, 0];
        let row_j = array![0, 1, 0, 1, 0, 1];
        let sign_i = 1;
        let sign_j = 1;
        let n_qubits = 3;

        let (new_p1, new_sign) = mult_paulis(&row_i, &row_j, sign_i, sign_j, n_qubits);

        assert_eq!(new_p1, array![1, 1, 1, 1, 1, 1]);
        assert_eq!(new_sign, 2);
    }

    #[test]
    fn test_mult_paulis_no_sign_change() {
        let row_i = array![1, 0, 0, 0, 1, 0];
        let row_j = array![0, 1, 0, 1, 0, 1];
        let sign_i = 0;
        let sign_j = 0;
        let n_qubits = 3;

        let (new_p1, new_sign) = mult_paulis(&row_i, &row_j, sign_i, sign_j, n_qubits);

        assert_eq!(new_p1, array![1, 1, 0, 1, 1, 1]);
        assert_eq!(new_sign, 0);
    }

    #[test]
    fn test_tableau_append_cnot() {
        let n_qubits = 3;

        let signs = array![0, 0, 0, 0, 0, 0];
        let tableau = get_default_tableau().clone();
        let mut ct = CliffordTableau { tableau, signs, n_qubits };

        ct.append_cnot(0, 1);

        let expected_tableau = array![
            [1, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ];

        assert_eq!(ct.tableau, expected_tableau);
        assert_eq!(ct.signs, array![0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_tableau_append_h() {
        let n_qubits = 3;
        let tableau = get_default_tableau().clone();
        let signs = array![0, 0, 0, 0, 0, 0];
        let mut ct = CliffordTableau { tableau, signs, n_qubits };

        ct.append_h(0);

        let expected_tableau = array![
            [0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ];

        assert_eq!(ct.tableau, expected_tableau);
        assert_eq!(ct.signs, array![0, 0, 0, 0, 0, 0]);
    }


    #[test]
    fn test_tableau_append_s() {
        let n_qubits = 3;
        let tableau = get_default_tableau().clone();
        let signs = array![0, 0, 0, 0, 0, 0];
        let mut ct = CliffordTableau { tableau, signs, n_qubits };

        ct.append_s(0);

        let expected_tableau = array![
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ];

        assert_eq!(ct.tableau, expected_tableau);
        assert_eq!(ct.signs, array![0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_tableau_prepend_cnot() {
        let n_qubits = 3;
        let signs = array![0, 0, 0, 0, 0, 0];
        let tableau = get_default_tableau().clone();
        let mut ct = CliffordTableau { tableau, signs, n_qubits };

        ct.append_cnot(0, 1);

        let expected_tableau = array![
            [1, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ];

        assert_eq!(ct.tableau, expected_tableau);
        assert_eq!(ct.signs, array![0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_tableau_prepend_h() {
        let n_qubits = 3;
        let tableau = get_default_tableau().clone();
        let signs = array![0, 0, 0, 0, 0, 0];
        let mut ct = CliffordTableau { tableau, signs, n_qubits };

        ct.append_h(0);

        let expected_tableau = array![
            [0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ];

        assert_eq!(ct.tableau, expected_tableau);
        assert_eq!(ct.signs, array![0, 0, 0, 0, 0, 0]);
    }


    #[test]
    fn test_tableau_prepend_s() {
        let n_qubits = 3;
        let tableau = get_default_tableau().clone();
        let signs = array![0, 0, 0, 0, 0, 0];
        let mut ct = CliffordTableau { tableau, signs, n_qubits };

        ct.append_s(0);

        let expected_tableau = array![
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ];

        assert_eq!(ct.tableau, expected_tableau);
        assert_eq!(ct.signs, array![0, 0, 0, 0, 0, 0]);
    }
}


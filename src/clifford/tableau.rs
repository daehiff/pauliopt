use ndarray::prelude::*;
use pyo3::prelude::*;

use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2};

fn mult_paulis(
    p1: &Array1<u8>,
    p2: &Array1<u8>,
    sign1: u8,
    sign2: u8,
    n_qubits: usize,
) -> (Array1<u8>, u8) {
    let mut x_1 = p1.slice(s![..n_qubits]).to_owned();
    let mut z_1 = p1.slice(s![n_qubits..]).to_owned();
    let x_2 = p2.slice(s![..n_qubits]).to_owned();
    let z_2 = p2.slice(s![n_qubits..]).to_owned();

    let mut x_1_z_2 = &z_1 * &x_2;
    let z_1_x_2 = &x_1 * &z_2;

    let ac = (&x_1_z_2 + &z_1_x_2) % 2;

    x_1 = (&x_1 + &x_2) % 2;
    z_1 = (&z_1 + &z_2) % 2;

    x_1_z_2 = ((&x_1_z_2 + &x_1 + &z_1) % 2) * &ac;
    let sign_change = ((ac.sum() + 2 * x_1_z_2.sum()) % 4) > 1;
    let new_sign = (sign1 + sign2 + sign_change as u8) % 4;
    let new_p1 = ndarray::concatenate![Axis(0), z_1, x_1];
    (new_p1, new_sign)
}

#[pyclass]
#[derive(Clone)]
pub struct CliffordTableau {
    n_qubits: usize,
    tableau: Array2<u8>,
    signs: Array1<u8>,
}

#[pymethods]
impl CliffordTableau {
    #[new]
    fn new(n_qubits: usize) -> Self {
        let tableau = Array::eye(2 * n_qubits); //.mapv(|x:u8| x != 0);
        let signs = Array::zeros(2 * n_qubits); //.mapv(|x:u8| x != 0);
        CliffordTableau {
            n_qubits: n_qubits,
            tableau: tableau,
            signs: signs,
        }
    }

    fn __str__(&self) -> PyResult<String> {
        let mut result = String::new();
        for i in 0..self.n_qubits {
            for j in 0..self.n_qubits {
                let destab = ["I", "X", "Z", "Y"][self.x_out(i, j) as usize];
                let stab = ["I", "X", "Z", "Y"][self.z_out(i, j) as usize];
                result.push_str(&destab);
                result.push_str("/");
                result.push_str(&stab);
                result.push_str(" ");
            }
            result.push_str("|");
            result.push_str(" ");
            let destab_sign = ["+", "-"][self.x_sign(i) as usize];
            let stab_sign = ["+", "-"][self.z_sign(i) as usize];
            result.push_str(&destab_sign);
            result.push_str("/");
            result.push_str(&stab_sign);
            result.push_str("\n");
        }

        Ok(result)
    }

    fn prepend_h(&mut self, qubit: usize) {
        let mut tmp = self.signs[self.n_qubits];
        self.signs[qubit] = self.signs[qubit+self.n_qubits];
        self.signs[qubit+self.n_qubits] = tmp;

        for i in 0..2*self.n_qubits {
            tmp = self.tableau[[qubit, i]];
            self.tableau[[qubit, i]] = self.tableau[[qubit + self.n_qubits, i]];
            self.tableau[[qubit + self.n_qubits, i]] = tmp;
        }
    }

    fn append_h(&mut self, qubit: usize) {
        for i in 0..2 * self.n_qubits {
            self.signs[i] = (self.signs[i]
                + self.tableau[[i, qubit]] * self.tableau[[i, qubit + self.n_qubits]])
                % 2;

            let tmp = self.tableau[[i, qubit]];
            self.tableau[[i, qubit]] = self.tableau[[i, qubit + self.n_qubits]];
            self.tableau[[i, qubit + self.n_qubits]] = tmp;
        }
    }

    unsafe fn prepend_s(&mut self, py: Python, qubit: usize) {
        let stabilizer = self.tableau.slice(s![qubit, ..]).to_owned();
        let destabilizer = self.tableau.slice(s![qubit + self.n_qubits, ..]).to_owned();
        let stab_sign = self.signs[qubit];
        let destab_sign = self.signs[qubit + self.n_qubits];

        let (_destabilizer, destab_sign) = mult_paulis(
            &stabilizer,
            &destabilizer,
            stab_sign,
            destab_sign,
            self.n_qubits,
        );
        self.insert_pauli_row(_destabilizer.into_pyarray(py), destab_sign, qubit)
    }

    fn append_s(&mut self, qubit: usize) {
        for i in 0..2 * self.n_qubits {
            self.signs[i] = (self.signs[i] + self.tableau[[i, qubit]] * self.tableau[[i, qubit + self.n_qubits]]) % 2;
            self.tableau[[i, qubit + self.n_qubits]] = (self.tableau[[i, qubit]] + self.tableau[[i, qubit + self.n_qubits]]) % 2;
            
        }
    }

    unsafe fn prepend_cnot(&mut self, py: Python, control: usize, target: usize) {
        let stabilizer_ctrl = self.tableau.slice(s![control, ..]).to_owned();
        let destabilizer_ctrl = self
            .tableau
            .slice(s![control + self.n_qubits, ..])
            .to_owned();
        let stab_sign_ctrl = self.signs[control];
        let destab_sign_ctrl = self.signs[control + self.n_qubits];

        let stabilizer_target = self.tableau.slice(s![target, ..]).to_owned();
        let destabilizer_target = self
            .tableau
            .slice(s![target + self.n_qubits, ..])
            .to_owned();
        let stab_sign_target = self.signs[target];
        let destab_sign_target = self.signs[target + self.n_qubits];

        let (stab_ctrl, stab_sign_ctrl) = mult_paulis(
            &stabilizer_ctrl,
            &stabilizer_target,
            stab_sign_ctrl,
            stab_sign_target,
            self.n_qubits,
        );

        let (destab_target, destab_sign_target) = mult_paulis(
            &destabilizer_target,
            &destabilizer_ctrl,
            destab_sign_target,
            destab_sign_ctrl,
            self.n_qubits,
        );

        self.insert_pauli_row(stab_ctrl.into_pyarray(py), stab_sign_ctrl, control);
        self.insert_pauli_row(
            destab_target.into_pyarray(py),
            destab_sign_target,
            target + self.n_qubits,
        )
    }

    fn append_cnot(&mut self, control: usize, target: usize) {
        for i in 0..2 * self.n_qubits {
            let x_ia = self.tableau[[i, control]];
            let x_ib = self.tableau[[i, target]];

            let z_ia = self.tableau[[i, control + self.n_qubits]];
            let z_ib = self.tableau[[i, target + self.n_qubits]];

            self.tableau[[i, target]] =
                (self.tableau[[i, target]] + self.tableau[[i, control]]) % 2;
            self.tableau[[i, control + self.n_qubits]] = (self.tableau
                [[i, control + self.n_qubits]]
                + self.tableau[[i, target + self.n_qubits]])
                % 2;

            let tmp_sum = (x_ib + z_ia + 1) % 2;
            self.signs[i] = (self.signs[i] + x_ia * z_ib * tmp_sum) % 2;
        }
    }



    unsafe fn insert_pauli_row(&mut self, pauli: &PyArray1<u8>, p_sign: u8, row: usize) {
        let _pauli = pauli.as_array();
        for i in 0..self.n_qubits {
            if (self.tableau[[row, i]] + &_pauli[i]) % 2 == 1 {
                self.tableau[[row, i]] = (self.tableau[[row, i]] + 1) % 2;
            }
            if (self.tableau[[row, i + self.n_qubits]] + &_pauli[i + self.n_qubits]) % 2 == 1 {
                self.tableau[[row, i + self.n_qubits]] = (self.tableau[[row, i + self.n_qubits]] + 1) % 2;
            }
        }
        if (self.signs[row] + p_sign) % 2 == 1 {
            self.signs[row] = (self.signs[row] + 1) % 2;
        }
    }

    fn x_sign(&self, row: usize) -> u8 {
        return self.signs[row] as u8;
    }

    fn z_sign(&self, row: usize) -> u8 {
        return self.signs[row + self.n_qubits] as u8;
    }

    fn x_out(&self, row: usize, col: usize) -> u8 {
        let x = self.tableau[[row, col]] as u8;
        let z = self.tableau[[row, col + self.n_qubits]] as u8;
        return x + 2 * z;
    }

    fn z_out(&self, row: usize, col: usize) -> u8 {
        let x = self.tableau[[row + self.n_qubits, col]] as u8;
        let z = self.tableau[[row + self.n_qubits, col + self.n_qubits]] as u8;
        return x + 2 * z;
    }

    fn get_tableau(&self, py: Python) -> Py<PyArray2<u8>> {
        let matrix = self.tableau.clone().mapv(|x: u8| x as u8);
        matrix.into_pyarray(py).to_owned()
    }

    fn get_signs(&self, py: Python) -> Py<PyArray1<u8>> {
        let matrix = self.signs.clone().mapv(|x: u8| x as u8);
        matrix.into_pyarray(py).to_owned()
    }
}

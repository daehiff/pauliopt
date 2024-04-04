import os
import pickle
package_directory = os.path.dirname(os.path.abspath(__file__))

# type molecule uccsd

def load_oplist(name, benchmark='uccsd'):
    if benchmark == 'molecule':
        fth = os.path.join(package_directory, 'data', name + '.pickle')
        with open(fth, 'rb') as f:
            entry = pickle.load(f)
        return entry
    if benchmark == 'uccsd':
        fth = "/Users/davidwinderl/Documents/Workspaces/Workspace/pauliopt/paulihedral/benchmark/data/BeH2_UCCSD.pickle"#os.path.join(package_directory, 'data', name + '_UCCSD.pickle')
        with open(fth, 'rb') as f:
            entry = pickle.load(f)
        return entry
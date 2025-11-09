from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
import py3Dmol

def recon_3d(data):
    mol = Chem.RWMol()

    # Add atoms
    for i, atom_features in enumerate(data.x):
        atomic_number = int(atom_features[0].item())
        atom = Chem.Atom(atomic_number)
        mol.AddAtom(atom)

    # Add bonds
    for src, dst in data.edge_index.t().tolist():
        if src < dst:
            mol.AddBond(int(src), int(dst), Chem.BondType.SINGLE)
    mol = mol.GetMol()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, pos in enumerate(data.pos):
        conf.SetAtomPosition(i, Point3D(pos[0].item(), pos[1].item(), pos[2].item()))
    mol.AddConformer(conf)
    name = data.name

    return mol
    
def viz_3d(mol):
    block = Chem.MolToPDBBlock(mol)
    view = py3Dmol.view(width=400, height=300)
    view.addModel(block, "pdb")
    view.setStyle({"stick": {}, "sphere": {"radius": 0.5}})
    view.zoomTo()
    view.show()
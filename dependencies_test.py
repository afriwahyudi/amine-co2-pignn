#!/usr/bin/env python3
"""
Complete dependencies test for PIGNN environment
Run this after completing your conda/pip installation steps
"""

import sys
import platform

def test_basic_python():
    """Test basic Python environment"""
    print("="*50)
    print("PYTHON ENVIRONMENT")
    print("="*50)
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    print()

def test_torch():
    """Test PyTorch installation and CUDA support"""
    print("="*50)
    print("PYTORCH & CUDA")
    print("="*50)
    
    try:
        import torch
        print(f"[OK] PyTorch version: {torch.__version__}")
        print(f"[OK] CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"[OK] CUDA version: {torch.version.cuda}")
            print(f"[OK] GPU device count: {torch.cuda.device_count()}")
            print(f"[OK] Current GPU: {torch.cuda.get_device_name(0)}")
            print(f"[OK] CUDNN version: {torch.backends.cudnn.version()}")
            
            # Test basic GPU operations
            device = torch.device('cuda')
            x = torch.randn(10, 10).to(device)
            y = torch.randn(10, 10).to(device)
            z = torch.mm(x, y)
            print(f"[OK] GPU tensor operations working")
        else:
            print("Warning! CUDA not available - running on CPU")
            
    except Exception as e:
        print(f"[NOT OK] Error with PyTorch: {e}")
    print()

def test_torch_geometric():
    """Test PyTorch Geometric and related packages"""
    print("="*50)
    print("PYTORCH GEOMETRIC ECOSYSTEM")
    print("="*50)
    
    # Test torch-geometric
    try:
        import torch_geometric
        print(f"[OK] PyTorch Geometric version: {torch_geometric.__version__}")
    except Exception as e:
        print(f"[NOT OK] Error importing torch_geometric: {e}")
    
    # Test torch-scatter
    try:
        import torch_scatter
        print(f"[OK] torch-scatter imported successfully")
    except Exception as e:
        print(f"[NOT OK] Error importing torch_scatter: {e}")
    
    # Test torch-sparse
    try:
        import torch_sparse
        print(f"[OK] torch-sparse imported successfully")
    except Exception as e:
        print(f"[NOT OK] Error importing torch_sparse: {e}")
    
    # Test torch-cluster
    try:
        import torch_cluster
        print(f"[OK] torch-cluster imported successfully")
    except Exception as e:
        print(f"[NOT OK] Error importing torch_cluster: {e}")
    
    # Test torch-spline-conv
    try:
        import torch_spline_conv
        print(f"[OK] torch-spline-conv imported successfully")
    except Exception as e:
        print(f"[NOT OK] Error importing torch_spline_conv: {e}")
    print()

def test_scientific_packages():
    """Test scientific computing packages"""
    print("="*50)
    print("SCIENTIFIC COMPUTING PACKAGES")  
    print("="*50)
    
    packages = [
        ('numpy', 'np'),
        ('pandas', 'pd'),
        ('matplotlib', 'plt'),
        ('seaborn', 'sns'),
        ('sklearn', 'sklearn'),
        ('networkx', 'nx'),
        ('plotly', 'px'),
        ('kaleido', 'kaleido'),
        ('optuna', 'optuna'),
    ]
    
    for package_name, alias in packages:
        try:
            if alias == 'plt':
                import matplotlib.pyplot as plt
                import matplotlib
                print(f"[OK] matplotlib version: {matplotlib.__version__}")
            elif alias == 'sklearn':
                import sklearn
                print(f"[OK] scikit-learn version: {sklearn.__version__}")
            elif package_name == 'kaleido':
                # Get kaleido version using importlib.metadata
                import importlib.metadata
                version = importlib.metadata.version('kaleido')
                print(f"[OK] kaleido version: {version}")
            elif package_name == 'optuna':
                import optuna
                print(f"[OK] optuna version: {optuna.__version__}")
            else:
                module = __import__(package_name, fromlist=[''])
                print(f"[OK] {package_name} version: {module.__version__}")
        except Exception as e:
            print(f"[NOT OK] Error importing {package_name}: {e}")
    print()

def test_chemistry_packages():
    """Test chemistry-related packages"""
    print("="*50)
    print("CHEMISTRY PACKAGES")
    print("="*50)
    
    # Test RDKit
    try:
        from rdkit import Chem
        from rdkit import __version__ as rdkit_version
        print(f"[OK] RDKit version: {rdkit_version}")
        
        # Test basic RDKit functionality
        mol = Chem.MolFromSmiles('CCO')
        if mol is not None:
            print("[OK] RDKit SMILES parsing working")
        else:
            print("Warning! RDKit SMILES parsing issue")
            
    except Exception as e:
        print(f"[NOT OK] Error importing RDKit: {e}")
    
    # Test py3Dmol
    try:
        import py3Dmol
        print(f"[OK] py3Dmol imported successfully")
    except Exception as e:
        print(f"[NOT OK] Error importing py3Dmol: {e}")
    print()

def test_ml_interpretability():
    """Test ML interpretability packages"""
    print("="*50)
    print("ML INTERPRETABILITY")
    print("="*50)
    
    try:
        import captum
        print(f"[OK] Captum version: {captum.__version__}")
        
        # Test basic captum functionality
        from captum.attr import IntegratedGradients
        print("[OK] Captum IntegratedGradients imported successfully")
        
    except Exception as e:
        print(f"[NOT OK] Error with Captum: {e}")
    print()

def test_jupyter():
    """Test Jupyter components"""
    print("="*50)
    print("JUPYTER ENVIRONMENT")
    print("="*50)
    
    try:
        import IPython
        print(f"[OK] IPython version: {IPython.__version__}")
    except Exception as e:
        print(f"[NOT OK] Error importing IPython: {e}")
    
    try:
        import ipykernel
        print(f"[OK] ipykernel imported successfully")
    except Exception as e:
        print(f"[NOT OK] Error importing ipykernel: {e}")
    print()

def test_basic_functionality():
    """Test basic PyTorch Geometric functionality"""
    print("="*50)
    print("BASIC FUNCTIONALITY TEST")
    print("="*50)
    
    try:
        import torch
        import torch_geometric
        from torch_geometric.data import Data
        
        # Create a simple graph
        edge_index = torch.tensor([[0, 1, 1, 2],
                                  [1, 0, 2, 1]], dtype=torch.long)
        x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index)
        print(f"[OK] Created PyG Data object: {data}")
        print(f"  - Number of nodes: {data.num_nodes}")
        print(f"  - Number of edges: {data.num_edges}")
        print(f"  - Node features shape: {data.x.shape}")
        
        # Test if CUDA is available and move data to GPU
        if torch.cuda.is_available():
            data = data.cuda()
            print(f"[OK] Successfully moved data to GPU")
            
    except Exception as e:
        print(f"[NOT OK] Error in basic functionality test: {e}")
    print()

def main():
    """Run all tests"""
    print("PIGNN ENVIRONMENT DEPENDENCY TEST")
    print("=" * 60)
    print("Testing all dependencies for your PIGNN environment...")
    print()
    
    test_basic_python()
    test_torch()
    test_torch_geometric()
    test_scientific_packages()
    test_chemistry_packages()
    test_ml_interpretability()
    test_jupyter()
    test_basic_functionality()
    
    print("="*60)
    print("DEPENDENCY TEST COMPLETED")
    print("="*60)
    print("If you see any [NOT OK] errors above, those packages may need reinstallation.")
    print("All [OK] marks indicate successful imports and basic functionality.")

if __name__ == "__main__":
    main()

from typing import Tuple, Sequence, Optional

import numpy as np

from rdkit import Chem

from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint
from mordred import Calculator, descriptors


morded_calculator = Calculator(descriptors, ignore_3D= False)


def mordred(smiles): return np.array([list(morded_calculator(Chem.MolFromSmiles(smiles))
                                      .fill_missing(value=0.)
                                      .values())])\
    .astype(np.float32)

def mordred_fp(smiles):
    fps = []
    for smile in smiles:
        fp = mordred(smile).reshape(-1)
        fps.append(fp)
    fps = np.array(fps)

    return fps

class MorganFPFeaturizer():

    def __init__(
        self,
        fp_size: int = 2048,
        radius: int = 2,
        use_counts: bool = False,
        use_features: bool = False,
        use_chirality=False,
        fingerprint_extra_args: Optional[dict] = None,
    ):
        """
        Args:
            fp_size: fingerprint length to generate.
            radius: fingerprint radius to generate.
            use_counts: use counts in fingerprint.
            use_features: use features in fingerprint.
            fingerprint_extra_args: kwargs for `GetMorganFingerprint`
        """
        super().__init__()

        if fingerprint_extra_args is None:
            fingerprint_extra_args = {}

        self.fp_size = fp_size
        self.radius = radius
        self.use_features = use_features
        self.use_counts = use_counts
        self.use_chirality = use_chirality
        self.fingerprint_extra_args = fingerprint_extra_args

    def transform_single(self, molecule: str) -> Tuple[np.ndarray, bool]:
        mol = Chem.MolFromSmiles(molecule.strip())

        if mol is None or len(molecule) == 0:
            return self.invalid_mol_features(), False

        return self.transform_mol(mol)

    def transform_mol(self, molecule: Chem.rdchem.Mol) -> Tuple[np.ndarray, bool]:
        use_chirality = self.__dict__.get('use_chirality', False)

        fp = GetMorganFingerprint(
            molecule,
            radius=self.radius,
            useFeatures=self.use_features,
            useCounts=self.use_counts,
            useChirality=use_chirality,
            **self.fingerprint_extra_args,
        )
        fp = rdkit_sparse_array_to_np(fp.GetNonzeroElements(
        ).items(), use_counts=self.use_counts, fp_size=self.fp_size)

        return fp, True

    def transform(self, molecules: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Featurizes a sequence of molecules.

        Args:
            molecules: molecules, given as a sequence of SMILES strings

        Returns:
            Tuple: 2D array for the feature vectors, 1D array for the validity masks
        """
        single_results = [self.transform_single(m) for m in molecules]
        features_list, mask_list = zip(*single_results)

        return np.vstack(features_list)

    def invalid_mol_features(self) -> np.ndarray:
        """
        Features to return for invalid molecules.
        """
        return np.zeros(self.output_size)

    @property
    def output_size(self) -> int:
        return self.fp_size


def rdkit_sparse_array_to_np(sparse_fp, use_counts, fp_size):
    """
    Converts an rdkit int hashed fingerprint into a 1D numpy array.

    Args:
        sparse_fp (dict: int->float): sparse dict of values set
        use_counts (bool): when folding up the hash, should it sum or not
        fp_size (int): length of fingerprint

    Returns:
        Numpy array of fingerprint
    """
    fp = np.zeros((fp_size,), np.int32)
    for idx, v in sparse_fp:
        if use_counts:
            fp[idx % fp_size] += int(v)
        else:
            fp[idx % fp_size] = 1

    return fp

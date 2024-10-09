import multiprocessing
import pickle
from pathlib import Path
from typing import Any, Union

import dgl
import torch
from rdkit import Chem

from molr.data_processing import mol_to_dgl
from molr.model import GNN


NUM_PROC = multiprocessing.cpu_count()


class MolRSmilesEmbedder:
    """Embeds SMILES strings using a pre-trained GNN model.
       The interface is designed to be used with the `datasets` library.
       Outputs zero vectors for invalid SMILES."""
    def __init__(
        self,
        model_path: Union[str, Path],
        device: Union[torch.device, None] = None,
    ):
        self.model_path = Path(model_path)
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self._load_model(self.model_path, self.device)

    def _load_model(self, model_path: Union[str, Path], device: torch.device) -> None:
        self.feature_encoder = pickle.load(
            model_path.joinpath('feature_enc.pkl').open('rb')
        )
        self.hparams = pickle.load(model_path.joinpath('hparams.pkl').open('rb'))
        self.embedder = GNN(
            self.hparams['gnn'], self.hparams['layer'],
            self.hparams['feature_len'], self.hparams['dim'])
        self.dim = self.hparams['dim']

        if torch.cuda.is_available():
            self.embedder.load_state_dict(
                torch.load(
                    model_path.joinpath('model.pt'),
                    map_location=torch.device('cpu'), weights_only=True))
            self.embedder = self.embedder.to(device)
        else:
            self.embedder.load_state_dict(
                torch.load(
                    model_path.joinpath('model.pt'),
                    map_location=torch.device('cpu'), weights_only=True))
        self.embedder.eval()

    @property
    def dimension(self) -> int:
        return self.dim

    def process_record(
        self,
        data: dict[str, Any],
        idx: Union[int, None] = None,
    ) -> dict[str, Any]:
        """Transforms data into an embedding.
           The data should contain a 'smiles' key with a SMILES string.

        Args:
            data (dict[str, Any]): dictionary containing the SMILES string.
                To obtain records from a pandas DataFrame, use
                `records = df.to_dict(orient='records')`.
            idx (Union[int, None], optional): index of the record in the
                dataset. If not None, a column `idx` will be added to features.
                Defaults to None.
        Returns:
            dict[str, Any]: dictionary containing the SMILES string
                and the 'vector' key with the embedding
        """
        device = next(self.embedder.parameters()).device
        graph = MolRSmilesEmbedder._smiles_to_dgl(
            data['smiles'], feature_encoder=self.feature_encoder
        )
        with torch.no_grad():
            if graph is not None:
                vector = self.embedder(graph.to(device)).squeeze(0).tolist()  # type: ignore
            else:
                vector = torch.zeros(self.dim, device=device).tolist()
        data['vector'] = vector
        if idx is not None:
            data['id'] = idx
        return data

    def process_batch(
        self,
        batch: dict[str, Any],
        indices: Union[list[int], None] = None,
        rank: Union[int, None] = None,
    ) -> dict[str, Any]:
        """Transforms SMILES into an embedding

        Args:
            batch (dict[str, Any]): record batch, dictionary containing the
                SMILES strings array in 'smiles' key. To obtain batch from
                a pandas DataFrame, use `batch = df.to_dict(orient='list')`.
            indices (Union[list[int], None], optional): indices for records
                in batch. If not None, a column `idx` will be added to features.
                Defaults to None.
            rank (Union[int, None], optional): MPI-style rank of the process.
                Can be used for multi-gpu parallelization. Defaults to None.

        Returns:
            dict[str, Any]: record batch containing the 'vector' key with the
                embeddings
        """
        device = next(self.embedder.parameters()).device
        graphs = [
            MolRSmilesEmbedder._smiles_to_dgl(
                smiles, feature_encoder=self.feature_encoder
            )
            for smiles in batch['smiles']
        ]
        valid_indices = [i for i, graph in enumerate(graphs) if graph is not None]
        graphs_gpu = [graph.to(device) for graph in graphs if graph is not None]
        with torch.no_grad():
            vectors = self.embedder(dgl.batch(graphs_gpu))
            if len(valid_indices) < len(graphs):
                vectors_full = torch.zeros((len(graphs), self.dim), device=device)
                vectors_full[valid_indices] = vectors
                batch['vector'] = vectors_full.tolist()
            else:
                batch['vector'] = vectors.tolist()
        if indices is not None:
            batch['idx'] = indices
        return batch

    @staticmethod
    def _smiles_to_dgl(
        smiles: str, feature_encoder: dict[str, Any]
    ) -> Union[dgl.DGLGraph, None]:
        """Converts a SMILES string into a DGLGraph

        Args:
            smiles (str): SMILES string
            feature_encoder (dict[str, Any]): feature encoder dictionary

        Returns:
            Union[dgl.DGLGraph, None]: DGLGraph object or None if the SMILES
                is invalid
        """
        try:
            mol = Chem.RemoveHs(Chem.MolFromSmiles(smiles))
            graph = mol_to_dgl(mol, feature_encoder)
        except Exception as e:
            graph = None
            print(f'Error in smiles_to_dgl_rdkit: {e}')
        return graph

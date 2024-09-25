import pickle
from pathlib import Path
from typing import Any

import dgl
import torch
from rdkit import Chem

from data_processing import mol_to_dgl
from model import GNN


class MolRSmilesEmbedder:
    def __init__(
        self,
        model_path: str | Path,
        device: torch.device | None = None,
        ):
        self.model_path = Path(model_path)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model(self.model_path, self.device)

    def load_model(self, model_path: str | Path, device: torch.device):
        self.feature_encoder = pickle.load(model_path.joinpath('feature_enc.pkl').open('rb'))
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


    def __call__(
        self, batch: dict[str, Any], indices: list[int] | None = None, rank: int | None = None):
        """Transforms SMILES into an embedding"""
        device = next(self.embedder.parameters()).device
        graphs = [self.smiles_to_dgl_rdkit(
            smiles, feature_encoder=self.feature_encoder) for smiles in batch['smiles']]
        valid_indices = [i for i, graph in enumerate(graphs) if graph is not None]
        graphs_gpu = [graph.to(device) for graph in graphs if graph is not None]
        with torch.no_grad():
            vectors = self.embedder(dgl.batch(graphs_gpu))
            if len(valid_indices) < len(graphs):
                vectors_full = torch.zeros((len(graphs), self.dim), device=device)
                vectors_full[valid_indices] = vectors
                batch['vector'] = vectors_full
            else:
                batch['vector'] = vectors
        return batch

    @staticmethod
    def smiles_to_dgl_rdkit(smiles: str, feature_encoder: dict[str, Any]) -> dgl.DGLGraph | None:
        try:
            mol = Chem.RemoveHs(Chem.MolFromSmiles(smiles))
            graph = mol_to_dgl(mol, feature_encoder)
        except Exception as e:
            graph = None
            print(f'Error in smiles_to_dgl_rdkit: {e}')
        return graph

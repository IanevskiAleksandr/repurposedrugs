import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, SAGEConv, global_mean_pool, BatchNorm
from torch_geometric.data import Data, DataLoader
from torch.nn import Sequential, Linear, ReLU, Dropout
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
import deepchem as dc
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

class DataProcessor:
    def __init__(self, file_path, problematic_smiles):
        self.df = pd.read_csv(file_path)
        self.problematic_smiles = problematic_smiles
        self.featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)

    def clean_data(self):
        self.df['smiles'] = self.df['smiles'].astype(str)
        self.df = self.df[self.df['smiles'] != 'nan']
        self.df['smiles'] = self.df['smiles'].str.strip()
        self.df = self.df[~self.df['smiles'].isin(self.problematic_smiles)]
        self.df['molecule'] = self.df['smiles'].apply(lambda x: Chem.MolFromSmiles(x) if x != 'nan' else None)
        self.df = self.df[self.df['molecule'].notnull()]
        self.df = self.df[self.df['molecule'].apply(lambda x: x is not None and x.GetNumAtoms() > 1)]
        self.df.drop(columns=['molecule'], inplace=True)
        self.df.fillna(self.df.median(numeric_only=True), inplace=True)
        self.df = pd.get_dummies(self.df, columns=['Disease_class', 'Disease_name'])

    def featurize_smiles(self):
        self.df['graph_features'] = self.df['smiles'].apply(self.safe_featurize)
        self.df = self.df.dropna(subset=['graph_features'])
        self.df = self.df.drop('graph_features', axis=1)

    def safe_featurize(self, smiles):
        try:
            features = self.featurizer.featurize([smiles])
            return None if features.size == 0 else features[0]
        except Exception as e:
            print(f"Failed to featurize {smiles}: {str(e)}")
            return None

    def smiles_to_graph(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        return self.mol_to_graph_data_obj(mol)

    @staticmethod
    def mol_to_graph_data_obj(mol):
        atom_features_list = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        x = torch.tensor(atom_features_list, dtype=torch.float).view(-1, 1)

        edge_index_list = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()]
        edge_index_list += [(j, i) for i, j in edge_index_list]
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

        data = Data(x=x, edge_index=edge_index)
        return data

    def preprocess_smiles_to_graph(self):
        self.df['graph_data'] = self.df['smiles'].apply(self.smiles_to_graph)

    def prepare_data(self):
        graph_data_list = []
        tabular_data_list = []
        targets = []

        for idx, row in self.df.iterrows():
            graph_data = row['graph_data']
            graph_data.batch = torch.zeros(graph_data.x.size(0), dtype=torch.long)
            graph_data_list.append(graph_data)

            tabular_features = row.drop(['Phase', 'smiles', 'graph_data']).values.astype(np.float32)
            tabular_data_list.append(torch.tensor(tabular_features, dtype=torch.float))

            targets.append(torch.tensor([row['Phase']], dtype=torch.float))

        return graph_data_list, tabular_data_list, targets

class CustomGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_layers, dropout_rate):
        super(CustomGNN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.convs.append(GINConv(Sequential(Linear(num_node_features, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))))
        self.bns.append(BatchNorm(hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm(hidden_channels))

        self.dropout = Dropout(dropout_rate)

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)

        x = global_mean_pool(x, batch)
        return x

class CNNTabularModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_filters, kernel_size, num_layers, dropout_rate):
        super(CNNTabularModel, self).__init__()
        self.convs = torch.nn.ModuleList()
        
        self.convs.append(torch.nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=kernel_size))

        for i in range(1, num_layers):
            in_channels = num_filters * (2 ** (i - 1))
            out_channels = num_filters * (2 ** i)
            self.convs.append(torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size))

        self.conv_output_size = self._get_conv_output(input_dim, kernel_size, num_layers, num_filters)

        self.fc1 = torch.nn.Linear(self.conv_output_size, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.dropout = Dropout(dropout_rate)

    def _get_conv_output(self, input_dim, kernel_size, num_layers, num_filters):
        output_dim = input_dim
        for _ in range(num_layers):
            output_dim = output_dim - (kernel_size - 1)
        return output_dim * (num_filters * (2 ** (num_layers - 1)))

    def forward(self, x):
        x = x.unsqueeze(1)
        for conv in self.convs:
            x = conv(x)
            x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class HybridModel(torch.nn.Module):
    def __init__(self, num_node_features, num_tabular_features, hidden_channels, hidden_dim, num_filters, kernel_size, num_gnn_layers, num_cnn_layers, dropout_rate):
        super(HybridModel, self).__init__()
        self.gnn = CustomGNN(num_node_features, hidden_channels, num_gnn_layers, dropout_rate)
        self.cnn_tabular = CNNTabularModel(num_tabular_features, hidden_dim, num_filters, kernel_size, num_cnn_layers, dropout_rate)
        self.fc1 = torch.nn.Linear(hidden_channels + hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, graph_data, tabular_data):
        x, edge_index, batch = graph_data.x, graph_data.edge_index, graph_data.batch
        x = self.gnn(x, edge_index, batch)
        y = self.cnn_tabular(tabular_data)
        combined = torch.cat([x, y], dim=1)
        combined = self.fc1(combined)
        combined = F.relu(combined)
        combined = self.fc2(combined)
        return combined

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, graph_data_list, tabular_data_list, targets):
        self.graph_data_list = graph_data_list
        self.tabular_data_list = tabular_data_list
        self.targets = targets

    def __len__(self):
        return len(self.graph_data_list)

    def __getitem__(self, idx):
        return self.graph_data_list[idx], self.tabular_data_list[idx], self.targets[idx]

class ModelTrainer:
    def __init__(self, df_filtered, params):
        self.df_filtered = df_filtered
        self.params = params
        self.num_node_features = 1
        self.num_tabular_features = len(df_filtered.columns) - 3
        self.kf = KFold(n_splits=10, shuffle=True, random_state=42)

    def train(self):
        fold_results = []
        out_of_fold_real_targets = []
        out_of_fold_predicted_targets = []

        for fold, (train_idx, test_idx) in enumerate(self.kf.split(self.df_filtered)):
            train_df = self.df_filtered.iloc[train_idx]
            test_df = self.df_filtered.iloc[test_idx]

            processor = DataProcessor(None, [])
            train_graph_data_list, train_tabular_data_list, train_targets = processor.prepare_data(train_df)
            test_graph_data_list, test_tabular_data_list, test_targets = processor.prepare_data(test_df)

            train_dataset = CustomDataset(train_graph_data_list, train_tabular_data_list, train_targets)
            test_dataset = CustomDataset(test_graph_data_list, test_tabular_data_list, test_targets)

            train_dataloader = DataLoader(train_dataset, batch_size=int(self.params['batch_size']), shuffle=True)
            test_dataloader = DataLoader(test_dataset, batch_size=int(self.params['batch_size']), shuffle=False)

            model = HybridModel(self.num_node_features, self.num_tabular_features, self.params['hidden_channels'], self.params['hidden_dim'],
                                self.params['num_filters'], self.params['kernel_size'], self.params['num_gnn_layers'],
                                self.params['num_cnn_layers'], self.params['dropout_rate'])
            optimizer = torch.optim.Adam(model.parameters(), lr=self.params['learning_rate'], weight_decay=self.params['weight_decay'])
            criterion = torch.nn.MSELoss()

            self._train_model(model, train_dataloader, test_dataloader, optimizer, criterion, fold_results, out_of_fold_real_targets, out_of_fold_predicted_targets)

        avg_test_loss_across_folds = np.mean(fold_results)
        correlation, _ = pearsonr(out_of_fold_real_targets, out_of_fold_predicted_targets)

        print(f"Parameters: {self.params}, Average Test Loss: {avg_test_loss_across_folds}, Pearson Correlation: {correlation}")

        return {'loss': avg_test_loss_across_folds, 'status': STATUS_OK, 'correlation': correlation}

    def _train_model(self, model, train_dataloader, test_dataloader, optimizer, criterion, fold_results, out_of_fold_real_targets, out_of_fold_predicted_targets):
        best_val_loss = float('inf')
        epochs_no_improve = 0
        early_stop = False

        for epoch in range(100):
            if early_stop:
                break

            model.train()
            epoch_loss = 0
            for data in train_dataloader:
                graph_data, tabular_data, target = data
                optimizer.zero_grad()
                out = model(graph_data, tabular_data)
                loss = criterion(out, target.view_as(out))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data in test_dataloader:
                    graph_data, tabular_data, target = data
                    out = model(graph_data, tabular_data)
                    loss = criterion(out, target.view_as(out))
                    val_loss += loss.item()

            val_loss /= len(test_dataloader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= 10:
                early_stop = True

        test_loss = 0
        fold_real_targets = []
        fold_predicted_targets = []
        with torch.no_grad():
            for data in test_dataloader:
                graph_data, tabular_data, target = data
                out = model(graph_data, tabular_data)
                loss = criterion(out, target.view_as(out))
                test_loss += loss.item()
                fold_real_targets.extend(target.view_as(out).cpu().numpy().flatten())
                fold_predicted_targets.extend(out.cpu().numpy().flatten())

        avg_test_loss = test_loss / len(test_dataloader)
        fold_results.append(avg_test_loss)

        out_of_fold_real_targets.extend(fold_real_targets)
        out_of_fold_predicted_targets.extend(fold_predicted_targets)

def objective(params):
    processor = DataProcessor("/data/phase_data.csv", problematic_smiles=['[Cl-].[K+]', '[As+3].[As+3].[O-2].[O-2].[O-2]', '[Cl-].[Na+]'])
    processor.clean_data()
    processor.featurize_smiles()
    processor.preprocess_smiles_to_graph()
    
    df_filtered = processor.df
    trainer = ModelTrainer(df_filtered, params)
    return trainer.train()

space = {
    'batch_size': hp.choice('batch_size', [8, 16, 32, 64]),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.01)),
    'dropout_rate': hp.uniform('dropout_rate', 0.1, 0.5),
    'hidden_dim': hp.choice('hidden_dim', [16, 32, 64, 128]),
    'hidden_channels': hp.choice('hidden_channels', [8, 16, 32, 64]),
    'num_filters': hp.choice('num_filters', [8, 16, 32, 64]),
    'kernel_size': hp.choice('kernel_size', [2, 3, 4, 5]),
    'num_gnn_layers': hp.choice('num_gnn_layers', [1, 2, 3, 4]),
    'num_cnn_layers': hp.choice('num_cnn_layers', [1, 2, 3, 4]),
    'weight_decay': hp.loguniform('weight_decay', np.log(1e-5), np.log(1e-3))
}

trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=200, trials=trials)

print(f'Best parameters: {best}')

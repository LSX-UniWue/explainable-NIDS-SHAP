
from tqdm import tqdm

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


class Autoencoder(torch.nn.Module):
    """
    Autoencoder being build with n_bottleneck neurons in bottleneck.
    Encoder and decoder contain n_layers each.
    size of layers starts at 2**(log2(n_bottleneck) + 1) near bottleneck and increases with 2**(last+1)
    """
    def __init__(self, n_inputs, cpus=0, n_layers=3, n_bottleneck=2**3, seed=0, **params):
        # setting number of threads for parallelization
        super(Autoencoder, self).__init__()

        torch.manual_seed(seed)
        if cpus > 0:
            torch.set_num_threads(cpus * 2)

        bottleneck_exp = (np.log2(n_bottleneck))

        # AE architecture
        layers = []
        # Input
        layers += [torch.nn.Linear(in_features=n_inputs,
                                   out_features=int(2**(bottleneck_exp + n_layers))),
                   torch.nn.ReLU()]
        # Encoder
        for i in range(n_layers - 1, 0, -1):  # layers from bottleneck: 8, 16, 32, 64, ...
            layers += [torch.nn.Linear(in_features=int(2**(bottleneck_exp + i + 1)),
                                       out_features=int(2**(bottleneck_exp + i))),
                       torch.nn.ReLU()]
        # Bottleneck
        layers += [torch.nn.Linear(in_features=int(2**(bottleneck_exp + 1)),
                                   out_features=n_bottleneck)]
        # Decoder
        for i in range(1, n_layers + 1):  # layers from bottleneck: 8, 16, 32, 64, ...
            layers += [torch.nn.Linear(in_features=int(2**(bottleneck_exp + i - 1)),
                                       out_features=int(2**(bottleneck_exp + i))),
                       torch.nn.ReLU()]
        # Output
        layers += [torch.nn.Linear(in_features=int(2**(bottleneck_exp + n_layers)),
                                   out_features=n_inputs)]  # output layer
        # Full model
        self.model = torch.nn.Sequential(*layers)
        self.add_module('model', self.model)
        self.add_module('distance_layer', module=torch.nn.PairwiseDistance(p=2))

        if 'learning_rate' in params:
            self.optim = torch.optim.Adam(params=self.model.parameters(), lr=params.pop('learning_rate'))
        else:
            self.optim = torch.optim.Adam(params=self.model.parameters())

        self.params = params

    def score_samples(self, x, output_to_numpy=True):
        x = self.to_tensor(x)
        loss = self.__call__(input_tensor=x)
        if output_to_numpy:
            return loss.data.numpy()
        else:
            return loss

    def to_tensor(self, x):
        if isinstance(x, pd.DataFrame):
            x = x.values
        if not torch.is_tensor(x):
            x = torch.Tensor(x)
        return x

    def reconstruct(self, x, output_to_numpy=True):
        x = self.to_tensor(x)
        out = self.model(x)
        if output_to_numpy:
            return out.data.numpy()
        else:
            return out

    def save(self, save_path):
        torch.save(self.state_dict(), save_path)

    def load(self, load_path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_state_dict(torch.load(load_path, map_location=device))
        return self

    def __call__(self, input_tensor, *args, **kwargs):
        pred = self.model(input_tensor)
        anom_score = self.distance_layer(pred, input_tensor)
        return anom_score  # the higher, the more abnormal (reconstruction error)

    def fit(self, data, device='cpu'):
        verbose = self.params['verbose']
        if isinstance(data, torch.utils.data.DataLoader):
            data_loader = data
        else:
            dataset = torch.utils.data.TensorDataset(torch.Tensor(data.values), torch.Tensor(data.values))
            data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                      batch_size=self.params['batch_size'],
                                                      shuffle=self.params['shuffle'])
        self.to(device)
        for _ in tqdm(range(self.params['epochs']), disable=verbose < 1):
            counter = 0
            for x, _ in tqdm(data_loader, disable=verbose < 2):
                x = x.to(device)
                y_pred = self.model(x)
                loss = self.distance_layer(y_pred, x).mean()

                self.model.zero_grad()
                loss.backward()
                self.optim.step()
                counter += 1

        return self

    def test(self, data, device='cpu', return_metrics=True):
        verbose = self.params['verbose']

        if isinstance(data, torch.utils.data.DataLoader):
            data_loader = data
        else:
            dataset = torch.utils.data.TensorDataset(torch.Tensor(data.values), torch.Tensor(data.values))
            data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                      batch_size=self.params['batch_size'],
                                                      shuffle=False)

        preds = []
        trues = []
        self.eval()
        self.to(device)

        for x, y in tqdm(data_loader, disable=verbose < 2):
            x = x.to(device)
            anom_score = self(x)
            preds.extend(list(anom_score.detach().cpu().numpy()))
            trues.extend(list(y.detach().cpu().numpy()))
        scores = np.array(preds)
        y = np.array(trues)

        if return_metrics:
            return {f'auc_pr': average_precision_score(y_true=y, y_score=scores),
                    f'auc_roc': roc_auc_score(y_true=y, y_score=scores)}

        else:
            return {'pred': scores, 'true': y}

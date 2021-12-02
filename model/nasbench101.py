import click
import warnings, os
from pathlib import Path
import nni
import nni.retiarii.evaluator.pytorch.lightning as pl
import torch.nn as nn
import torch
import torchmetrics
from nni.retiarii import model_wrapper, serialize, serialize_cls
from nni.retiarii.nn.pytorch import NasBench101Cell
from timm.optim import RMSpropTF
from torch.optim.lr_scheduler import CosineAnnealingLR

from .base_ops import Conv3x3BnRelu, Conv1x1BnRelu, Projection


@model_wrapper
class NasBench101(nn.Module):
    def __init__(self,
                 stem_in_channels: int = 3,
                 stem_out_channels: int = 128,
                 num_stacks: int = 3,
                 num_modules_per_stack: int = 3,
                 max_num_vertices: int = 7,
                 max_num_edges: int = 9,
                 num_labels: int = 10,
                 bn_eps: float = 1e-5,
                 bn_momentum: float = 0.003):
        super().__init__()

        op_candidates = {
            'conv3x3-bn-relu': lambda num_features: Conv3x3BnRelu(num_features, num_features),
            'conv1x1-bn-relu': lambda num_features: Conv1x1BnRelu(num_features, num_features),
            'maxpool3x3': lambda num_features: nn.MaxPool2d(3, 1, 1)
        }

        # initial stem convolution
        self.stem_conv = Conv3x3BnRelu(stem_in_channels, stem_out_channels)

        layers = []
        in_channels = out_channels = stem_out_channels
        for stack_num in range(num_stacks):
            if stack_num > 0:
                downsample = nn.MaxPool2d(kernel_size=2, stride=2)
                layers.append(downsample)
                out_channels *= 2
            for _ in range(num_modules_per_stack):
                cell = NasBench101Cell(op_candidates, in_channels, out_channels,
                                       lambda cin, cout: Projection(cin, cout),
                                       max_num_vertices, max_num_edges, label='cell')
                layers.append(cell)
                in_channels = out_channels

        self.features = nn.ModuleList(layers)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(out_channels, num_labels)

        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eps = bn_eps
                module.momentum = bn_momentum

    def forward(self, x):
        bs = x.size(0)
        out = self.stem_conv(x)
        for layer in self.features:
            out = layer(out)
        out = self.gap(out).view(bs, -1)
        out = self.classifier(out)
        return out

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eps = self.config.bn_eps
                module.momentum = self.config.bn_momentum


class AccuracyWithLogits(torchmetrics.Accuracy):
    def update(self, pred, target):
        return super().update(nn.functional.softmax(pred), target)


@serialize_cls
class NasBench101TrainingModule(pl.LightningModule):
    def __init__(self, max_epochs=108, learning_rate=0.001, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters('learning_rate', 'weight_decay', 'max_epochs')
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = AccuracyWithLogits()
        export_onnx = True
        if export_onnx is None or export_onnx is True:
            self.export_onnx = Path(os.environ.get('NNI_OUTPUT_DIR', '.')) / 'model.onnx'
            self.export_onnx.parent.mkdir(exist_ok=True)
        elif export_onnx:
            self.export_onnx = Path(export_onnx)
        else:
            self.export_onnx = None
        self._already_exported = False

    def forward(self, x):
        y_hat = self.model(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_accuracy', self.accuracy(y_hat, y), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        if not self._already_exported:
            try:
                self.to_onnx(self.export_onnx, x, export_params=True)
            except RuntimeError as e:
                warnings.warn(f'ONNX conversion failed. As a result, you might not be able to use visualization. Error message: {e}')
            self._already_exported = True
        
        self.log('val_loss', self.criterion(y_hat, y), prog_bar=True)
        self.log('val_accuracy', self.accuracy(y_hat, y), prog_bar=True)

    def configure_optimizers(self):
        # optimizer = RMSpropTF(self.parameters(), lr=self.hparams.learning_rate,
        #                       weight_decay=self.hparams.weight_decay,
        #                       momentum=0.9, alpha=0.9, eps=1.0)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate,
                              weight_decay=self.hparams.weight_decay)
        return {
            'optimizer': optimizer,
            'scheduler': CosineAnnealingLR(optimizer, self.hparams.max_epochs)
        }

    def on_validation_epoch_end(self):
        nni.report_intermediate_result(self.trainer.callback_metrics['val_accuracy'].item())

    def teardown(self, stage):
        if stage == 'fit':
            nni.report_final_result(self.trainer.callback_metrics['val_accuracy'].item())



import click
import nni.retiarii.evaluator.pytorch.lightning as pl
from nni.retiarii import model_wrapper, serialize, serialize_cls
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, KMNIST, Caltech101, FashionMNIST

from model.nni_utils import MapElitesStrategy
from model.nasbench101 import NasBench101TrainingModule, NasBench101
from map_elites import EliteMatrix

@click.command()
@click.option('--epochs', default=20, help='Training length.')
@click.option('--batch_size', default=128, help='Batch size.')
@click.option('--port', default=8081, help='On which port the experiment is run.')
@click.option('--dataset', default='KMNIST')
@click.option('--benchmark', is_flag=True, default=False)
def _multi_trial_test(epochs, batch_size, port, dataset, benchmark):
    # initalize dataset. Note that 50k+10k is used. It's a little different from paper
    transf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip()
    ]
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.25]) if dataset in ['MNIST', 'KMNIST', 'FashionMNIST'] else\
        transforms.Normalize([0.49139968, 0.48215827, 0.44653124], [0.24703233, 0.24348505, 0.26158768]) 
    ]
    train_dataset = serialize(eval(dataset), f'/mnt/workspace/DATASET/{dataset}', train=True, download=True, transform=transforms.Compose(transf + normalize))
    test_dataset = serialize(eval(dataset), f'/mnt/workspace/DATASET/{dataset}', train=False, transform=transforms.Compose(normalize))

    # specify training hyper-parameters
    training_module = NasBench101TrainingModule(max_epochs=epochs)
    # FIXME: need to fix a bug in serializer for this to work
    # lr_monitor = serialize(LearningRateMonitor, logging_interval='step')
    trainer = pl.Trainer(max_epochs=epochs, gpus=1)
    lightning = pl.Lightning(
        lightning_module=training_module,
        trainer=trainer,
        train_dataloader=pl.DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        val_dataloaders=pl.DataLoader(test_dataset, batch_size=batch_size)
    )

    elite_matrix = EliteMatrix.load(f'nasbench_gt_{dataset}.npy')
    strategy = MapElitesStrategy(elite_matrix, metric_name=dataset)

    model = NasBench101(
        stem_in_channels = 1 if dataset in ['MNIST', 'KMNIST', 'FashionMNIST'] else 3,
        num_labels=101 if dataset == 'Caltech101' else 10
    )

    exp = RetiariiExperiment(model, lightning, [], strategy)

    exp_config = RetiariiExeConfig('local')
    exp_config.trial_concurrency = 1
    exp_config.max_trial_number = 100
    exp_config.trial_gpu_number = 1 
    exp_config.training_service.use_active_gpu = True
    # exp_config.execution_engine = 'base'

    if benchmark:
        exp_config.benchmark = 'nasbench101'
        exp_config.execution_engine = 'benchmark'

    exp.run(exp_config, port)


if __name__ == '__main__':
    _multi_trial_test()




# from nni.retiarii.converter import convert_to_graph


# script_module = torch.jit.script(base_model)
# base_model_ir = convert_to_graph(script_module, base_model)
# codegen.model_to_pytorch_script(model)

# graph_data = BaseGraphData.load(receive_trial_parameters())
# random_str = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
# file_name = f'_generated_model/{random_str}.py'
# os.makedirs(os.path.dirname(file_name), exist_ok=True)
# with open(file_name, 'w') as f:
#     f.write(graph_data.model_script)
# model_cls = utils.import_(f'_generated_model.{random_str}._model')
# graph_data.evaluator._execute(model_cls)
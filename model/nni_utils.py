import time
from nni.retiarii.strategy.base import BaseStrategy
from nni.retiarii import InvalidMutation, submit_models, query_available_resources, budget_exhausted, wait_models, ModelStatus
from nni.retiarii.strategy.bruteforce import _logger
from nni.retiarii.strategy.utils import dry_run_for_search_space, get_targeted_model, filter_model


class MapElitesStrategy(BaseStrategy):
    """
    Random search on the search space.

    Parameters
    ----------
    variational : bool
        Do not dry run to get the full search space. Used when the search space has variational size or candidates. Default: false.
    dedup : bool
        Do not try the same configuration twice. When variational is true, deduplication is not supported. Default: true.
    model_filter: Callable[[Model], bool]
        Feed the model and return a bool. This will filter the models in search space and select which to submit.
    """

    def __init__(self, elites, model_filter=None, metric_name="new_metric"):
        self.elites = elites
        self.filter = model_filter
        self.metric_name = metric_name

    def get_sample(self, data, search_space):
        adj = data["module_adjacency"]
        module_operations = data["module_operations"]
        
        num_node = len(module_operations)
        sample = {}
        for space, k in search_space.items():
            if space[0].label == 'cell/num_nodes':
                assert num_node in k
                sample[space] = num_node
            elif space[0].label.startswith('cell/op'):
                node_id = int(space[0].label[-1])
                # Hack to ignore some 
                if node_id >= num_node - 1:
                    sample[space] = 'maxpool3x3'
                else:
                    sample[space] = module_operations[node_id]
            elif space[0].label.startswith('cell/input'):
                from_id = space[1]
                to_id = int(space[0].label[-1])
                # 
                if from_id >= num_node or to_id >= num_node:
                    sample[space] = False
                    continue
                if adj[from_id][to_id]:
                    sample[space] = True
                else:
                    sample[space] = False
        return sample
        

    def run(self, base_model, applied_mutators):
        _logger.info('Search in Elites: self.elites')
        search_space = dry_run_for_search_space(base_model, applied_mutators)
        print(search_space,'\n')
        print(search_space.keys(),'\n')
        print(base_model)
        # import ipdb; ipdb.set_trace()
        for space, k in search_space.items():
            print(space[0].label, k)
        # import ipdb; ipdb.set_trace()

        # TODO: code here manage samples
        submited_models = []
        for elite_property, elite in self.elites.sorted_iter():
            _logger.debug('New model created. Waiting for resource. %s', str(elite))
            if self.metric_name in elite.keys():
                _logger.info(f'Jumping over elite: {elite}')
                continue

            # Block if no availabe resources 
            while query_available_resources() <= 0:
                wait_models(*[x[0] for x in submited_models])
                for model, e, ep in submited_models:
                    if model.status == ModelStatus.Failed:
                        raise Exception
                    rew = model.metric
                    self.elites.update_metric(rew, ep, key=self.metric_name)
                    _logger.info(f'Model metric received as reward: {rew}')
                    print(self.elites.get_performance(key=self.metric_name))
                self.elites.save(f'./nasbench_gt_{self.metric_name}.npy')
                # if budget_exhausted():
                #     return
                # time.sleep(self._polling_interval)

            try:
                sample = self.get_sample(elite, search_space)
                model = get_targeted_model(base_model, applied_mutators, sample)
                if filter_model(self.filter, model):
                    submit_models(model)
                    submited_models.append((model, elite, elite_property))
                    # Hack for single thread program, might want to 
                    
            except InvalidMutation as e:
                _logger.warning(f'Invalid mutation: {e}. Skip.')
            self.elites.save(f'./nasbench_gt_{self.metric_name}.npy')
            

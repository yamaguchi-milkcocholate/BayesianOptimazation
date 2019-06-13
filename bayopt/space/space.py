from GPyOpt.core.task.space import Design_space


def initialize_space(domain, constraints):
    if not isinstance(domain, list):
        raise ValueError('domain has to be a list')

    for idx, dim in enumerate(domain):
        dim['name'] = str(idx)

    return Design_space(space=domain, constraints=constraints)


def get_subspace(space, subspace_idx):
    subspace_domain = [dim for idx, dim in enumerate(space.config_space) if idx in subspace_idx]
    # Todo: constraints
    return Design_space(space=subspace_domain, constraints=None)

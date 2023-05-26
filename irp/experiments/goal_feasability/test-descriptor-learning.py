import irp.experiments.tile_coding.env as env
import irp.experiments.tile_coding.q as q
import irp.envs as envs
import irp.wrappers as wrappers
import irp.utils

def get_data(train_name, test_name, parameters):
    subimage_width = parameters['subimage_width']
    subimage_height = parameters['subimage_height']
    overlap = parameters['overlap']

    # Get all the subimages
    (train_Xs, train_ys), (test_Xs, test_ys) = irp.utils.make_sample_label(
        train_name, test_name, width=subimage_width, height=subimage_height, overlap=overlap, idx=None
    )

    return train_Xs, train_ys, test_Xs, test_ys

def filter_data(Xs, ys, coordinate, parameters):
    return irp.utils.get_neighborhood_images(
        Xs, ys, coordinate,
        subimage_width=parameters['subimage_width'],
        subimage_height=parameters['subimage_height'],
        overlap=parameters['overlap'],
        n_size=parameters['n_size'],
        neighborhood=parameters['neighborhood']
    )

def make_environment(Xs, ys, parameters):
    environments = wrappers.MultiSample([])

    for subimage, sublabel in zip(Xs, ys):
        environment = env.Env(subimage, sublabel, parameters['n_thresholds'])
        environment = wrappers.Tiled(
            environment,
            tiles_per_dim=parameters['tiles_per_dim'],
            tilings=parameters['tilings'],
            limits=parameters['limits']
        )

    return environments

train_name = 'case10_10.png'
test_name = 'case10_11.png'

parameters = {
    'subimage_width': 8,
    'subimage_height': 8,
    'overlap': 0.0,
    'n_thresholds': 5,
    'n_size': 1,
    'neighborhood': 'neumann',
    'tiles_per_dim': (2, 2, 2),
    'tilings': 16,
    'limits': [(0, 1), (0, 1), (0, 32)]
}

# Load the training and testing subimage
train_Xs, train_ys, test_Xs, test_ys = get_data(train_name, test_name, parameters)

# Filter the training data on neighborhood subimages for a specific coordinate of interest only
coordinate = (272, 216)
nh_Xs, nh_ys = filter_data(train_Xs, train_ys, coordinate, parameters)

# Set-up the MultiSample environment
environments = make_environment(nh_Xs, nh_ys, parameters)

q.learn(environments, log=True)


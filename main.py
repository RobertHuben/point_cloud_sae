import torch
import random
import time

from point_cloud_sae import TopkSAE, SAEAnthropic
from point_cloud_datasets import generate_datasets_for_saes
from analysis import graph_point_cloud_results, graph_point_cloud_results_unified, graph_reconstruction_errors

if __name__=="__main__":
    # make datasets
    modes_with_specs={'basic_blobs':5, 'lollipops':5, 'blob_grid':18, 'random_blobs':10}
    mode='basic_blobs'
    num_classes=modes_with_specs[mode]
    num_anchors=80
    train_size=100
    test_size=1000
    class_weights_seed = 1 #randomizes class weights if >0
    anchors, train_dataset, test_dataset=generate_datasets_for_saes(mode, num_anchors, train_size, test_size, num_classes, class_weights_seed=class_weights_seed)

    # other details for training
    num_epochs=1000000//train_size
    cloud_scale_factor=10
    custom_number_of_features=None
    if custom_number_of_features==None:
        num_features=num_classes
    else:
        num_features=custom_number_of_features

    for seed in [3]:
        # anchors, train_dataset, test_dataset=generate_datasets_for_saes(mode, num_anchors, train_size, test_size, num_classes, class_weights_seed=class_weights_seed, other_random_seed=seed)

        torch.random.manual_seed(seed)
        sae=SAEAnthropic(anchors, num_features=num_features, l1_sparsity_coefficient=5, cloud_scale_factor=cloud_scale_factor)
        # sae=TopkSAE(anchors, num_features=num_features, k=1, use_relu=True, l1_sparsity_coefficient=0, cloud_scale_factor=cloud_scale_factor)

        sae.train_sae(train_dataset=train_dataset, eval_dataset=test_dataset, learning_rate=1e-4, num_epochs=num_epochs, report_every_n_data=10000, batch_size=100)
    torch.save(sae, "trained_saes/point_cloud_sae.pkl")

    graph_point_cloud_results(test_dataset, anchors=anchors)
    graph_reconstruction_errors(test_dataset)
    graph_point_cloud_results_unified(test_dataset, anchors=anchors, anchor_details='encoder', save_name='all_features_encoder')
    graph_point_cloud_results_unified(test_dataset, anchors=anchors, anchor_details='decoder', save_name='all_features_decoder')
    test_dataset.plot_as_scatter(labels=sae.classify(test_dataset), save_name="classfications")
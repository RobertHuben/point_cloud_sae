import torch
import random
import time

from point_cloud_sae import TopkSAE, SAEAnthropic, graph_point_cloud_results, graph_point_cloud_results_unified
from point_cloud_datasets import PointCloudDataset, generate_point_cloud_blobs, generate_point_cloud_lollipops 


if __name__=="__main__":
    mode='blobs'
    # mode='lollipops'
    num_anchors=100
    train_size=100
    test_size=1000
    num_epochs=100000//train_size
    num_features = 3 if mode == "blobs" else 5

    randomize_class_weights=False
    if randomize_class_weights:
        class_weights=[random.choice([1,4]) for _ in range(num_features)]
    else:
        class_weights=[1 for _ in range(num_features)]

    if mode=='blobs':
        anchors=generate_point_cloud_blobs(num_anchors, seed=1, class_weights=class_weights)
        train_dataset=PointCloudDataset(generate_point_cloud_blobs(train_size, seed=2, class_weights=class_weights))
        test_dataset=PointCloudDataset(generate_point_cloud_blobs(test_size, seed=3, class_weights=class_weights))
    elif mode=='lollipops':
        anchors=generate_point_cloud_lollipops(num_anchors, seed=1, class_weights=class_weights)
        train_dataset=PointCloudDataset(generate_point_cloud_lollipops(train_size, seed=2, class_weights=class_weights))
        test_dataset=PointCloudDataset(generate_point_cloud_lollipops(test_size, seed=3, class_weights=class_weights))

    sae=SAEAnthropic(anchors, num_features=num_features, l1_sparsity_coefficient=2, cloud_scale_factor=5)
    # sae=TopkSAE(anchors, num_features=num_features, k=1, use_relu=True, l1_sparsity_coefficient=0, cloud_scale_factor=5)
    t_start=time.time()
    sae.train_sae(train_dataset=train_dataset, eval_dataset=test_dataset, learning_rate=1e-4, num_epochs=num_epochs, report_every_n_data=5000, batch_size=100)
    t_end=time.time()
    print(f"{t_end-t_start} seconds")
    torch.save(sae, "trained_saes/point_cloud_sae.pkl")
    graph_point_cloud_results(test_dataset, anchors=anchors)
    graph_point_cloud_results_unified(test_dataset, anchors=anchors, anchor_details='encoder', save_name='all_features_encoder')
    graph_point_cloud_results_unified(test_dataset, anchors=anchors, anchor_details='decoder', save_name='all_features_decoder')
    print(f"Class weights are: {class_weights}")
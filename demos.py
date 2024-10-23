import torch
import math
import matplotlib.pyplot as plt

from point_cloud_datasets import generate_datasets_for_saes
from point_cloud_sae import TopkSAE, TrainingLog
from analysis import plot_many_training_runs

modes_with_num_classes={'basic_blobs':5, 'lollipops':5, 'blob_grid':18, 'random_blobs':10}

def basic_demo_on_target_dataset(mode):
    plt.figure(figsize=(12,5))
    num_classes=modes_with_num_classes[mode]
    num_anchors=100
    train_size=1000
    test_size=1000

    num_epochs=500000//train_size
    cloud_scale_factor=2/num_classes
    num_features=num_classes

    training_logs=[]
    saes=[]
    num_runs=10

    for seed in range(10):
        anchors, train_dataset, test_dataset=generate_datasets_for_saes(mode, num_anchors, train_size, test_size, 
                                                                        num_classes, class_weights_seed=seed, 
                                                                        other_random_seed=seed)
        torch.random.manual_seed(seed)
        sae=TopkSAE(anchors, num_features=num_features, k=1, use_relu=True, l1_sparsity_coefficient=0, cloud_scale_factor=cloud_scale_factor)
        train_log=TrainingLog()
        sae.train_sae(train_dataset=train_dataset, eval_dataset=test_dataset, learning_rate=1e-4, num_epochs=num_epochs, report_every_n_data=10000, batch_size=100, write_location=train_log)
        training_logs.append(train_log)
        saes.append(sae)
    plot_many_training_runs(training_logs, title=f'{num_runs} training runs on {mode} dataset', save_name_suffix=mode, save=False, close_at_end=False, num_suplots=3)
    entropies=[tl.entropies[-1] for tl in training_logs]
    median_entropy=sorted(entropies)[len(entropies)//2]
    median_seed=entropies.index(median_entropy)
    median_sae=saes[median_seed]
    _, __, final_test_dataset=generate_datasets_for_saes(mode, num_anchors, train_size, test_size, 
                                                                        num_classes, class_weights_seed=median_seed, 
                                                                        other_random_seed=median_seed)
    plt.subplot(1,3,3)
    final_test_dataset.plot_as_scatter(labels=median_sae.classify(final_test_dataset), close_at_end=False)
    plt.title(f"Clusters of median-entropy run\nSeed {median_seed}, Entropy: {median_entropy:.3f}")

    plt.tight_layout()
    plt.savefig(f"analysis_results/10_seeds_experiment{mode}.png")



if __name__=="__main__":
    for mode in modes_with_num_classes:
        basic_demo_on_target_dataset(mode)
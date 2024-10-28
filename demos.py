import numpy as np
import torch
import math
import matplotlib.pyplot as plt

from point_cloud_datasets import generate_datasets_for_saes
from point_cloud_sae import TopkSAE, TrainingLog
from analysis import plot_many_training_runs, graph_one_feature
from utils import embed_point_cloud

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

    for seed in range(num_runs):
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
    if mode in ['blob_grid']:
        plt.legend().remove()

    plt.tight_layout()
    plt.savefig(f"analysis_results/10_seeds_experiment{mode}.png")

def cloud_scale_sensitivity_test(mode='basic_blobs'):
    plt.figure(figsize=(12,5))
    num_classes=modes_with_num_classes[mode]
    num_anchors=100
    train_size=1000
    test_size=1000

    base_cloud_scale_factor=2/num_classes
    cloud_scale_multipliers=[x/10 for x in range(1,31)]
    cloud_scale_factors=[base_cloud_scale_factor*cloud_scale_multiplier for cloud_scale_multiplier in cloud_scale_multipliers]

    num_training_steps=500000
    num_epochs=num_training_steps//train_size
    num_features=num_classes

    training_logs=[]
    saes=[]
    num_runs=10

    for cloud_scale_factor in cloud_scale_factors:
        these_logs=[]
            
        for seed in range(num_runs):
            anchors, train_dataset, test_dataset=generate_datasets_for_saes(mode, num_anchors, train_size, test_size, 
                                                                            num_classes, class_weights_seed=seed, 
                                                                            other_random_seed=seed)
            torch.random.manual_seed(seed)
            sae=TopkSAE(anchors, num_features=num_features, k=1, use_relu=True, l1_sparsity_coefficient=0, cloud_scale_factor=cloud_scale_factor)
            train_log=TrainingLog()
            sae.train_sae(train_dataset=train_dataset, eval_dataset=test_dataset, learning_rate=1e-4, num_epochs=num_epochs, report_every_n_data=num_training_steps, batch_size=100, write_location=train_log)
            these_logs.append(train_log)
            saes.append(sae)
        training_logs.append(these_logs)
    data=[torch.tensor([train_log.entropies[-1] for train_log in x]) for x in training_logs]
    plt.boxplot(data)

    tick_labels=[f"{x:.2f}" for x in cloud_scale_factors]
    plt.xticks(range(1,len(cloud_scale_factors)+1), tick_labels)
    plt.xlabel(f"Cloud scale factor ({base_cloud_scale_factor:.2f} is default)")
    plt.ylabel("Entropy")
    plt.title(f"Variable Sweep: Cloud scale factor\n{mode} dataset, 10 runs each")
    plt.tight_layout()
    plt.savefig(f"analysis_results/cluster_scale_sensitivity_test_{mode}.png")
    plt.close()


def data_scarcity_test(mode='basic_blobs'):
    plt.figure(figsize=(12,5))
    num_classes=modes_with_num_classes[mode]
    test_size=1000
    num_anchors=100

    cloud_scale_factor=2/num_classes
    num_training_steps=500000
    num_features=num_classes

    training_logs=[]
    saes=[]
    num_runs=10

    data_sizes=[x*10 for x in range(1,21)]

    for data_size in data_sizes:
        train_size=data_size
        num_epochs=num_training_steps//train_size

        these_logs=[]
            
        for seed in range(num_runs):
            anchors, train_dataset, test_dataset=generate_datasets_for_saes(mode, num_anchors, train_size, test_size, 
                                                                            num_classes, class_weights_seed=seed, 
                                                                            other_random_seed=seed)
            torch.random.manual_seed(seed)
            sae=TopkSAE(anchors, num_features=num_features, k=1, use_relu=True, l1_sparsity_coefficient=0, cloud_scale_factor=cloud_scale_factor)
            train_log=TrainingLog()
            sae.train_sae(train_dataset=train_dataset, eval_dataset=test_dataset, learning_rate=1e-4, num_epochs=num_epochs, report_every_n_data=num_training_steps, batch_size=100, write_location=train_log)
            these_logs.append(train_log)
            saes.append(sae)
        training_logs.append(these_logs)
    data=[torch.tensor([train_log.entropies[-1] for train_log in x]) for x in training_logs]
    plt.boxplot(data)

    plt.plot([0, len(data_sizes)], [0.1,0.1], linestyle='dashed', c='black')
    plt.plot([0, len(data_sizes)], [0,0], c='black')
    tick_labels=[f"{x}" for x in data_sizes]
    plt.xticks(range(1,len(data_sizes)+1), tick_labels)
    plt.xlabel(f"Size of training dataset.")
    plt.ylabel("Entropy")
    plt.title(f"Variable Sweep: Dataset Size\n{mode} dataset, 10 runs each")
    plt.tight_layout()
    plt.savefig(f"analysis_results/data_scarcity_test_{mode}.png")
    plt.close()

def graph_all_true_classes():
    for n, mode in enumerate(modes_with_num_classes):
        ax=plt.subplot(2,2,n+1)
        num_classes=modes_with_num_classes[mode]
        _, point_cloud, __=generate_datasets_for_saes(mode, 1, 1000, 1, num_classes, class_weights_seed=-1, points_seeds=[0,1,0])
        point_cloud.plot_as_scatter(save_name=None, close_at_end=False)
        ax.get_legend().remove()
        plt.title(mode)
    plt.suptitle("Datasets used, with true clusters shown")
    plt.tight_layout()
    plt.savefig("analysis_results/true_clusters_plot.png")
    plt.close()

def num_features_test(mode='basic_blobs'):
    plt.figure(figsize=(12,5))
    num_classes=modes_with_num_classes[mode]
    train_size=1000
    test_size=1000
    num_anchors=100

    cloud_scale_factor=2/num_classes
    num_training_steps=500000

    num_features_values=[x+num_classes for x in range(-3, 4)]

    training_logs=[]
    saes=[]
    num_runs=10

    for num_features in num_features_values:
        num_epochs=num_training_steps//train_size

        these_logs=[]
            
        for seed in range(num_runs):
            anchors, train_dataset, test_dataset=generate_datasets_for_saes(mode, num_anchors, train_size, test_size, 
                                                                            num_classes, class_weights_seed=seed, 
                                                                            other_random_seed=seed)
            torch.random.manual_seed(seed)
            sae=TopkSAE(anchors, num_features=num_features, k=1, use_relu=True, l1_sparsity_coefficient=0, cloud_scale_factor=cloud_scale_factor)
            train_log=TrainingLog()
            sae.train_sae(train_dataset=train_dataset, eval_dataset=test_dataset, learning_rate=1e-4, num_epochs=num_epochs, report_every_n_data=num_training_steps, batch_size=100, write_location=train_log)
            these_logs.append(train_log)
            saes.append(sae)
        training_logs.append(these_logs)
    data=[torch.tensor([train_log.test_losses[-1] for train_log in x]) for x in training_logs]
    plt.boxplot(data)

    tick_labels=[f"{x}" for x in num_features_values]
    plt.xticks(range(1,len(num_features_values)+1), tick_labels)
    plt.plot([0, len(num_features_values)], [0,0], c='black')

    plt.xlabel(f"Number of SAE features\n(Number of classes: {num_classes})")
    plt.ylabel("Test loss")
    plt.title(f"Variable Sweep: n_features\n{mode} dataset, 10 runs each")
    plt.tight_layout()
    plt.savefig(f"analysis_results/number_of_features_test_{mode}.png")
    plt.close()

def graph_one_anchor_influence(activation_cutoff=1e-1):
    mode='basic_blobs'
    num_anchors=100
    train_size=1
    test_size=1000
    num_classes=5
    anchor_to_use=82

    anchors, train_dataset, test_dataset=generate_datasets_for_saes(mode, num_anchors, train_size, test_size, 
                                                                num_classes, class_weights_seed=-1, 
                                                                other_random_seed=-1)

    anchors=anchors[anchor_to_use,:].unsqueeze(0)

    x=test_dataset.points[:,0].detach().numpy()
    y=test_dataset.points[:,1].detach().numpy()
    embeddings=embed_point_cloud(test_dataset.points, anchors)
    colors=embeddings.detach().numpy()
    plt.scatter(x,y, c=colors, label="Point embeddings")
    plt.clim(0,1)
    plt.colorbar(ticks=np.arange(0,11,1)/10)
    anchor_x=anchors[:,0].detach().numpy()
    anchor_y=anchors[:,1].detach().numpy()
    plt.scatter(anchor_x, anchor_y, c='red', marker='^', linewidths=0.5, edgecolors='black', label="Anchor")
    plt.legend()
    plt.title("Point embeddings from one anchor")
    plt.tight_layout()
    plt.savefig("analysis_results/point_embeddings_from_anchor.png")


def advanced_demo_on_target_dataset(mode):
    plt.figure(figsize=(12,5))
    num_classes=modes_with_num_classes[mode]
    num_anchors=1000
    train_size=1000
    test_size=1000

    num_epochs=500000//train_size
    cloud_scale_factor=2/num_classes
    num_features=num_classes

    training_logs=[]
    saes=[]
    num_runs=10

    for seed in range(num_runs):
        anchors, train_dataset, test_dataset=generate_datasets_for_saes(mode, num_anchors, train_size, test_size, 
                                                                        num_classes, class_weights_seed=seed, 
                                                                        other_random_seed=seed)
        torch.random.manual_seed(seed)
        sae=TopkSAE(anchors, num_features=num_features, k=1, use_relu=True, l1_sparsity_coefficient=0, cloud_scale_factor=cloud_scale_factor)
        train_log=TrainingLog()
        sae.train_sae(train_dataset=train_dataset, eval_dataset=test_dataset, learning_rate=1e-4, num_epochs=num_epochs, report_every_n_data=10000, batch_size=100, write_location=train_log)
        training_logs.append(train_log)
        saes.append(sae)
    plot_many_training_runs(training_logs, title=f'{num_runs} training runs on {mode} dataset', save_name_suffix=mode, save=False, close_at_end=False, num_suplots=4, use_adjoint_entropies=True)
    entropies=[tl.entropies[-1] for tl in training_logs]
    adjoint_entropies=[tl.adjoint_entropies[-1] for tl in training_logs]
    median_entropy=sorted(entropies)[len(entropies)//2]
    median_seed=entropies.index(median_entropy)
    median_adjoint_entropy=adjoint_entropies[median_seed]
    median_sae=saes[median_seed]
    _, __, final_test_dataset=generate_datasets_for_saes(mode, num_anchors, train_size, test_size, 
                                                                        num_classes, class_weights_seed=median_seed, 
                                                                        other_random_seed=median_seed)
    adjoint_clusterings=median_sae.adjoint_classify(final_test_dataset)

    plt.subplot(1,4,3)
    final_test_dataset.plot_as_scatter(labels=median_sae.classify(final_test_dataset), close_at_end=False)
    plt.legend().remove()
    plt.title(f"Clusters \nfrom median-entropy run\nSeed {median_seed}, Entropy: {median_entropy:.3f}")

    plt.subplot(1,4,4)
    final_test_dataset.plot_as_scatter(labels=adjoint_clusterings, close_at_end=False)
    plt.legend().remove()
    plt.title(f"Adjoint Clusters \n from median-entropy run\nSeed {median_seed}, Entropy: {median_adjoint_entropy:.3f}")


    plt.tight_layout()
    plt.savefig(f"analysis_results/10_seeds_experiment_advanced_{mode}.png")


if __name__=="__main__":
    # for mode in modes_with_num_classes:
        # num_features_test(mode)
        # data_scarcity_test(mode)
        # cloud_scale_sensitivity_test(mode)
    # num_features_test('random_blobs')
    # num_features_test('basic_blobs')
    # data_scarcity_test('basic_blobs')
    # data_scarcity_test('random_blobs')
    # cloud_scale_sensitivity_test('basic_blobs')
    # cloud_scale_sensitivity_test('random_blobs')
    for mode in ['blob_grid']:
        basic_demo_on_target_dataset(mode)
        # advanced_demo_on_target_dataset(mode)
    # graph_all_true_classes()
    # graph_one_anchor_influence()

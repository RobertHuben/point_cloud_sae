import torch
import matplotlib.pyplot as plt
import os
from point_cloud_sae import SAETemplate, SAEAnthropic, TopkSAE

from math import ceil, sqrt

torch.serialization.add_safe_globals([SAETemplate, SAEAnthropic])
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def graph_point_cloud_results_unified(eval_dataset, save_name=None, anchors=None, file_location="trained_saes/point_cloud_sae.pkl", detail_level=3, activation_cutoff=1e-1, anchor_details='encoder'):
    sae=torch.load(file_location)
    number_of_features=sae.num_features
    num_subplots_horizontal=ceil(sqrt(number_of_features))
    num_subplots_vertical=ceil(number_of_features/num_subplots_horizontal)

    for i in range(number_of_features):
        ax=plt.subplot(num_subplots_vertical, num_subplots_horizontal, i+1)
        graph_one_feature(sae, i, eval_dataset, axes=ax, anchors=anchors, detail_level=detail_level, activation_cutoff=activation_cutoff, anchor_details=anchor_details)
    plt.tight_layout()
    save_directory="analysis_results"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    if not save_name:
        save_name="all_features"
    save_file_name=f"{save_directory}/{save_name}_detail_{detail_level}.png"
    plt.suptitle(f"All feature activations\nwith {anchor_details} weights shown")
    plt.tight_layout()

    plt.savefig(save_file_name)
    plt.close()

def graph_reconstruction_errors(eval_dataset, file_location="trained_saes/point_cloud_sae.pkl", save_directory="analysis_results"):
    sae=torch.load(file_location)
    x=eval_dataset.points[:,0].detach().numpy()
    y=eval_dataset.points[:,1].detach().numpy()
    residual_streams, hidden_layers, reconstructed_residual_streams=sae.catenate_outputs_on_dataset(eval_dataset, include_loss=False)
    errors=(reconstructed_residual_streams-residual_streams).norm(dim=1)
    errors=errors.detach().numpy()
    point_cloud_scatter=plt.scatter(x,y, c=errors, label="Point errors")
    plt.title(f"Reconstruction Errors")
    plt.colorbar(point_cloud_scatter)
    save_file_name=f"{save_directory}/reconstruction_errors.png"
    plt.tight_layout()
    plt.savefig(save_file_name)
    plt.close()


def graph_one_feature(sae, feature_number, eval_dataset, axes, anchors=None, detail_level=3, activation_cutoff=1e-1, anchor_details='encoder'):
    x=eval_dataset.points[:,0].detach().numpy()
    y=eval_dataset.points[:,1].detach().numpy()
    residual_streams, hidden_layers, reconstructed_residual_streams=sae.catenate_outputs_on_dataset(eval_dataset, include_loss=False)
    hidden_layers=hidden_layers.detach().numpy()
    this_hidden_layers=hidden_layers[:,feature_number]
    this_hidden_layers_masked=this_hidden_layers*(this_hidden_layers>activation_cutoff)
    if detail_level==0:
        colors='blue'
        sizes=12
    else:
        colors=this_hidden_layers_masked
        sizes=18* (this_hidden_layers_masked>0)+6
    point_cloud_scatter=axes.scatter(x,y, c=colors, s=sizes, label="Point activations")
    axes.set_title(f"Activations of feature {feature_number}")
    if anchors != None:
        anchor_x=anchors[:,0].detach().numpy()
        anchor_y=anchors[:,1].detach().numpy()
        if detail_level==2:
            anchors_scatter= axes.scatter(anchor_x, anchor_y, c='r', marker='^')
        elif detail_level==3:
            if anchor_details=="encoder":
                anchor_weights=sae.encoder[:,feature_number]
            elif anchor_details=="decoder":
                anchor_weights=sae.decoder[feature_number]
            colors=anchor_weights.detach().numpy()
            sizes=torch.abs(anchor_weights).detach().numpy()
            eps=1e-6
            sizes=sizes/(sizes.max()+eps)*50 + 6
            anchors_scatter= axes.scatter(anchor_x, anchor_y, c=colors, marker='^', s=sizes, cmap='seismic', linewidths=0.5, edgecolors='black', label="Anchor influence on feature")
    else:
        anchors_scatter=None
    return point_cloud_scatter, anchors_scatter

def graph_point_cloud_results(eval_dataset, anchors=None, file_location="trained_saes/point_cloud_sae.pkl", detail_level=3, activation_cutoff=1e-1):
    '''
    by detail level:
    0 - just the points
    1 - points with feature activations
    2 - also anchor locations
    3 - also anchor directions
    '''
    sae=torch.load(file_location)
    save_directory="analysis_results"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    for feature_number in range(sae.num_features):
        fig, ax=plt.subplots()
        point_cloud_scatter, anchors_scatter=graph_one_feature(sae, feature_number, eval_dataset, axes=ax, anchors=anchors, detail_level=detail_level, activation_cutoff=activation_cutoff)
        plt.colorbar(point_cloud_scatter)
        plt.legend()
        if detail_level==3 and anchors_scatter:
            plt.colorbar(anchors_scatter)
        save_file_name=f"{save_directory}/feature_{feature_number}_detail_{detail_level}.png"
        plt.tight_layout()
        plt.savefig(save_file_name)
        plt.close()
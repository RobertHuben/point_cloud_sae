import os
import torch
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from tqdm import tqdm
from math import sqrt, ceil
from torch.nn import functional as F

from point_cloud_datasets import PointCloudDataset
from utils import embed_point_cloud, entropy_from_counts

device='cuda' if torch.cuda.is_available() else 'cpu'

class SAETemplate(torch.nn.Module, ABC):
    '''
    abstract base class that defines the SAE contract
    '''

    def __init__(self, anchor_points:torch.tensor, num_features:int, decoder_initialization_scale: float, cloud_scale_factor=1):
        super().__init__()
        self.anchor_points=anchor_points
        self.num_features=num_features
        self.cloud_scale_factor=cloud_scale_factor
        self.encoder, self.encoder_bias, self.decoder, self.decoder_bias=self.create_linear_encoder_decoder(decoder_initialization_scale)
        self.num_data_trained_on=0

    def catenate_outputs_on_dataset(self, dataset:PointCloudDataset, batch_size=8, include_loss=False):
        '''
        runs the model on the entire dataset, one batch at a time, catenating the outputs
        '''
        losses=[]
        residual_streams=[]
        hidden_layers=[]
        reconstructed_residual_streams=[]
        test_dataloader=iter(torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False))
        for test_input in test_dataloader:
            test_input=test_input.to(device)
            loss, residual_stream, hidden_layer, reconstructed_residual_stream = self.forward_on_points(test_input, compute_loss=include_loss)
            if include_loss:
                losses.append(loss)
            residual_streams.append(residual_stream)
            hidden_layers.append(hidden_layer)
            reconstructed_residual_streams.append(reconstructed_residual_stream)
        residual_streams=torch.cat(residual_streams)
        hidden_layers=torch.cat(hidden_layers)
        reconstructed_residual_streams=torch.cat(reconstructed_residual_streams)
        if include_loss:
            losses=torch.stack(losses)
            return losses, residual_streams, hidden_layers, reconstructed_residual_streams
        else:
            return residual_streams, hidden_layers, reconstructed_residual_streams
        
    def compute_l0_sparsity(self, hidden_layers):
        active_features=hidden_layers>0
        sparsity_per_entry=active_features.sum()/hidden_layers[..., 0].numel()
        return sparsity_per_entry

    def find_dead_features(self, hidden_layers, eps=1e-6):
        active_features=hidden_layers>eps
        dead_features=torch.all(torch.flatten(~ active_features, end_dim=-2), dim=0)
        return dead_features
    
    def count_dead_features(self, hidden_layers, eps=1e-6):
        dead_features=self.find_dead_features(hidden_layers, eps=eps)
        return dead_features.sum()

    def compute_classification_entropy(self, hidden_layers:torch.tensor,  eval_dataset:PointCloudDataset):
        cluster_assignments=hidden_layers.argmax(dim=1)
        return eval_dataset.compute_entropy_of_clustering(cluster_assignments)

    def print_evaluation(self, train_loss, eval_dataset:PointCloudDataset, step_number="N/A"):
        losses, residual_streams, hidden_layers, reconstructed_residual_streams=self.catenate_outputs_on_dataset(eval_dataset, include_loss=True)
        test_loss=losses.mean()
        l0_sparsity=self.compute_l0_sparsity(hidden_layers)
        dead_features=self.count_dead_features(hidden_layers)
        entropy=self.compute_classification_entropy(hidden_layers, eval_dataset)
        print_message=f"Train loss, test loss, l0 sparsity, dead features, entropy after training on {self.num_data_trained_on} datapoints: {train_loss.item():.2f}, {test_loss:.2f}, {l0_sparsity:.1f}, {dead_features:.0f}, {entropy:.2f}"
        tqdm.write(print_message)
        
    def forward_on_points(self, point_batch:torch.tensor, compute_loss=False):
        embedded_points=embed_point_cloud(point_batch, anchors=self.anchor_points, scale_factor=self.cloud_scale_factor)
        return self.forward(embedded_points, compute_loss=compute_loss)

    def train_sae(self, train_dataset:PointCloudDataset, eval_dataset:PointCloudDataset, batch_size=64, num_epochs=1, report_every_n_data=500, reinitialize_every_n_data=500, learning_rate=1e-3, fixed_seed=1337):
        '''
        performs a training loop on self, with printed evaluations
        '''
        if fixed_seed:
            torch.manual_seed(fixed_seed)
        self.to(device)
        self.train()
        optimizer=torch.optim.AdamW(self.parameters(), lr=learning_rate)
        step=0
        report_on_batch_number=report_every_n_data//batch_size
        reinitialize_on_batch_number=reinitialize_every_n_data//batch_size

        self.training_prep(train_dataset=train_dataset, eval_dataset=eval_dataset, batch_size=batch_size, num_epochs=num_epochs)

        print(f"Beginning model training on {device}!")

        for epoch in tqdm(range(num_epochs)):
            train_dataloader=iter(torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True))
            
            for input_batch in train_dataloader:
                input_batch=input_batch.to(device)
                step+=1
                self.num_data_trained_on+=len(input_batch)
                optimizer.zero_grad(set_to_none=True)
                loss, residual_stream, hidden_layer, reconstructed_residual_stream= self.forward_on_points(input_batch, compute_loss=True)
                loss.backward()
                optimizer.step()

                if step % report_on_batch_number==0:
                    self.print_evaluation(loss, eval_dataset, step_number=step)
                # if step % reinitialize_on_batch_number==0:
                #     self.check_for_and_reinitialize_dead_feature(train_dataset)
                self.after_step_update(train_dataset)
        else:
            self.print_evaluation(train_loss=loss, eval_dataset=eval_dataset, step_number="Omega")
        self.eval()

    @abstractmethod
    def forward(self, embedded_points, compute_loss=False):
        pass

    def create_linear_encoder_decoder(self, decoder_initialization_scale):
        # embedding_size=len(self.anchor_points)
        embedding_size=embed_point_cloud(self.anchor_points, anchors=self.anchor_points,eps=0.01, scale_factor=self.cloud_scale_factor).shape[1]
        decoder_initial_value=torch.randn((self.num_features, embedding_size))
        decoder_initial_value=decoder_initial_value/decoder_initial_value.norm(dim=1).unsqueeze(-1) # columns of norm 1
        decoder_initial_value*=decoder_initialization_scale # columns of norm decoder_initial_value
        encoder=torch.nn.Parameter(torch.clone(decoder_initial_value).transpose(0,1).detach())
        encoder_bias=torch.nn.Parameter(torch.zeros((self.num_features)))
        decoder=torch.nn.Parameter(decoder_initial_value)
        decoder_bias=torch.nn.Parameter(torch.zeros((embedding_size)))
        return encoder, encoder_bias, decoder, decoder_bias
    
    def training_prep(self, train_dataset=None, eval_dataset=None, batch_size=None, num_epochs=None):
        '''
        for anything additional that needs to be done before training starts
        '''
        return
    
    def after_step_update(self, train_dataset, hidden_layer=None, step=None):
        '''
        for anything additional that needs to be done after each training step
        '''
        losses, residual_streams, hidden_layers, reconstructed_residual_streams=self.catenate_outputs_on_dataset(train_dataset, include_loss=True)
        dead_features=self.find_dead_features(hidden_layers)
        if torch.any(dead_features):
            self.do_ghost_grads(dead_features, residual_streams, hidden_layer, reconstructed_residual_streams)

    def do_ghost_grads(self,dead_features,residual_streams, hidden_layer, reconstructed_residual_streams):
        learning_rate=1e-4
        errors=residual_streams-reconstructed_residual_streams
        error_magnitudes=errors.norm(dim=-1)**2
        mean_error_magnitude=error_magnitudes.mean()
        error_weighted_activation_direction=((residual_streams*error_magnitudes.unsqueeze(1)).sum(dim=0))/error_magnitudes.sum()
        error_weighted_error_direction=((errors*error_magnitudes.unsqueeze(1)).sum(dim=0))/error_magnitudes.sum()
        with torch.no_grad():
            self.encoder[:,dead_features]+=mean_error_magnitude*learning_rate*error_weighted_activation_direction.unsqueeze(1)
            self.decoder[dead_features,:]+=mean_error_magnitude*learning_rate*error_weighted_error_direction.unsqueeze(0)

    def reconstruction_error(self, residual_stream, reconstructed_residual_stream):
        reconstruction_l2=torch.norm(reconstructed_residual_stream-residual_stream, dim=-1)
        reconstruction_loss=(reconstruction_l2**2).mean()
        return reconstruction_loss
    


class SAEAnthropic(SAETemplate):

    def __init__(self, anchor_points:torch.tensor, num_features:int, l1_sparsity_coefficient:float, decoder_initialization_scale=0.1, cloud_scale_factor=1):
        super().__init__(anchor_points=anchor_points, num_features=num_features, decoder_initialization_scale=decoder_initialization_scale, cloud_scale_factor=cloud_scale_factor)
        self.l1_sparsity_coefficient=l1_sparsity_coefficient

    def forward(self, embedded_points, compute_loss=False):
        hidden_layer=self.activation_function(embedded_points @ self.encoder + self.encoder_bias)
        reconstructed_embedded_points=hidden_layer @ self.decoder + self.decoder_bias
        if compute_loss:
            loss = self.loss_function(embedded_points, hidden_layer, reconstructed_embedded_points)
        else:
            loss = None
        return loss, embedded_points, hidden_layer, reconstructed_embedded_points
    
    def loss_function(self, residual_stream, hidden_layer, reconstructed_residual_stream):
        reconstruction_loss=self.reconstruction_error(residual_stream, reconstructed_residual_stream)
        sparsity_loss= self.sparsity_loss_function(hidden_layer)*self.l1_sparsity_coefficient
        total_loss=reconstruction_loss+sparsity_loss
        return total_loss

    def activation_function(self, encoder_output):
        return F.relu(encoder_output)

    def sparsity_loss_function(self, hidden_layer):
        decoder_column_norms=self.decoder.norm(dim=1)
        return torch.mean(hidden_layer*decoder_column_norms)
    
    def report_model_specific_features(self):
        return [f"Sparsity loss coefficient: {self.l1_sparsity_coefficient}"]



#suppression_mode can be "relative" or "absolute"
class LeakyTopkSAE(SAETemplate):
    def __init__(self, anchor_points:torch.tensor, num_features: int, leakiness: float, k:int, suppression_mode="relative", decoder_initialization_scale=0.1, use_relu=True, l1_sparsity_coefficient=0, cloud_scale_factor=1):
        super().__init__(anchor_points=anchor_points, num_features=num_features, decoder_initialization_scale=decoder_initialization_scale, cloud_scale_factor=cloud_scale_factor)
        self.leakiness = leakiness
        self.k=k
        self.suppression_mode = suppression_mode
        self.use_relu=use_relu
        self.l1_sparsity_coefficient=l1_sparsity_coefficient


    def activation_function(self, encoder_output):
        if self.use_relu:
            activations = torch.relu(encoder_output)
        else:
            activations=encoder_output
        kth_value = torch.topk(activations, k=self.k).values.min(dim=-1).values
        return suppress_lower_activations(activations, kth_value, leakiness=self.leakiness, mode=self.suppression_mode)
    
    def forward(self, residual_stream, compute_loss=False):
        '''
        takes the trimmed residual stream of a language model (as produced by run_gpt_and_trim) and runs the SAE
        must return a tuple (loss, residual_stream, hidden_layer, reconstructed_residual_stream)
        residual_stream is shape (B, W, D), where B is batch size, W is (trimmed) window length, and D is the dimension of the model:
            - residual_stream is unchanged, of size (B, W, D)
            - hidden_layer is of shape (B, W, D') where D' is the size of the hidden layer
            - reconstructed_residual_stream is shape (B, W, D) 
        '''
        normalized_encoder = F.normalize(self.encoder, p=2, dim=1) #normalize columns
        # normalized_decoder = normalized_encoder.T
        normalized_decoder = F.normalize(self.decoder, p=2, dim=1) #normalize columns
        hidden_layer=self.activation_function((residual_stream - self.decoder_bias) @ normalized_encoder + self.encoder_bias)
        reconstructed_residual_stream=hidden_layer @ normalized_decoder + self.decoder_bias
        loss= self.reconstruction_error(residual_stream, reconstructed_residual_stream) if compute_loss else None
        return loss, residual_stream, hidden_layer, reconstructed_residual_stream

    def report_model_specific_features(self):
        return [f"k (sparsity): {self.k}", f"Leakiness: {self.leakiness}"]

    def loss_function(self, residual_stream, hidden_layer, reconstructed_residual_stream):
        reconstruction_loss=self.reconstruction_error(residual_stream, reconstructed_residual_stream)
        sparsity_loss= self.sparsity_loss_function(residual_stream)*self.l1_sparsity_coefficient
        total_loss=reconstruction_loss+sparsity_loss
        return total_loss

    def sparsity_loss_function(self, residual_stream):
        normalized_encoder = F.normalize(self.encoder, p=2, dim=1) #normalize columns
        normalized_decoder = F.normalize(self.decoder, p=2, dim=1) #normalize columns
        pre_activation_hidden_layer=(residual_stream - self.decoder_bias) @ normalized_encoder + self.encoder_bias
        decoder_column_norms=self.decoder.norm(dim=1)
        return torch.mean(pre_activation_hidden_layer*decoder_column_norms)

class TopkSAE(LeakyTopkSAE):
    def __init__(self, anchor_points:torch.tensor, num_features:int, k:int, decoder_initialization_scale=0.1, use_relu=True, l1_sparsity_coefficient=0, cloud_scale_factor=1):
        super().__init__(anchor_points=anchor_points, 
                        num_features=num_features, 
                        leakiness=0, 
                        k=k, 
                        suppression_mode="relative", 
                        decoder_initialization_scale=decoder_initialization_scale, 
                        use_relu=use_relu, 
                        l1_sparsity_coefficient=l1_sparsity_coefficient,
                        cloud_scale_factor=cloud_scale_factor)


def suppress_lower_activations(t, bound, leakiness, inclusive=True, mode="absolute"):
    if torch.is_tensor(bound) and bound.numel() != 1:
        while bound.dim() < t.dim():
            bound = torch.unsqueeze(bound, -1)
    above_mask = (torch.abs(t) >= bound) if inclusive else (torch.abs(t) > bound)
    above_only = t * above_mask
    below_only = t * (~above_mask)
    if mode == "absolute":
        bad_bound_mask = bound <= 0 #to make sure we don't divide by 0
        return above_only + (~bad_bound_mask)*leakiness/(bound+bad_bound_mask) * below_only
    elif mode == "relative":
        return above_only + leakiness * below_only


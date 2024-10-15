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

    def __init__(self, anchor_points:torch.tensor, 
                    num_features:int, 
                    decoder_initialization_scale: float,
                    cloud_scale_factor:float=1, 
                    l1_sparsity_coefficient:float=0,
                    ghost_loss_coefficient:float=1,
                    ghost_loss_style="new"):
        super().__init__()
        self.anchor_points=anchor_points
        self.num_features=num_features
        self.cloud_scale_factor=cloud_scale_factor
        self.encoder, self.encoder_bias, self.decoder, self.decoder_bias=self.create_linear_encoder_decoder(decoder_initialization_scale)
        self.num_data_trained_on=0
        self.ghost_loss_style=ghost_loss_style
        self.num_data_since_last_activation=torch.zeros((num_features)) #used for ghost grads
        self.l1_sparsity_coefficient=l1_sparsity_coefficient
        self.ghost_loss_coefficient=ghost_loss_coefficient


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
        '''
        reutrns a tensor of shape (num_features), which is True if the feature is dead, and False otherwise
        '''
        active_features=hidden_layers>eps
        dead_features=torch.all(torch.flatten(~ active_features, end_dim=-2), dim=0)
        return dead_features
    
    def count_dead_features(self, hidden_layers, eps=1e-6):
        dead_features=self.find_dead_features(hidden_layers, eps=eps)
        return dead_features.sum()

    def compute_classification_entropy(self, hidden_layers:torch.tensor,  eval_dataset:PointCloudDataset):
        cluster_assignments=hidden_layers.argmax(dim=1)
        return eval_dataset.compute_entropy_of_clustering(cluster_assignments)

    @torch.inference_mode()
    def print_evaluation(self, train_loss, eval_dataset:PointCloudDataset, step_number="N/A"):
        losses, residual_streams, hidden_layers, reconstructed_residual_streams=self.catenate_outputs_on_dataset(eval_dataset, include_loss=True)
        test_loss=losses.mean()
        l0_sparsity=self.compute_l0_sparsity(hidden_layers)
        dead_features=self.count_dead_features(hidden_layers)
        entropy=self.compute_classification_entropy(hidden_layers, eval_dataset)
        print_message=f"Train loss, test loss, l0 sparsity, dead features, entropy after training on {self.num_data_trained_on} datapoints: {train_loss.item():.2f}, {test_loss:.2f}, {l0_sparsity:.1f}, {dead_features:.0f}, {entropy:.2f}"
        tqdm.write(print_message)
        
    def forward_on_points(self, point_batch:torch.tensor, compute_loss=False) -> torch.tensor:
        embedded_points=embed_point_cloud(point_batch, anchors=self.anchor_points, scale_factor=self.cloud_scale_factor)
        return self.forward(embedded_points, compute_loss=compute_loss)

    def train_sae(self, train_dataset:PointCloudDataset, eval_dataset:PointCloudDataset, batch_size=64, num_epochs=1, report_every_n_data=500, learning_rate=1e-3, fixed_seed=1337):
        '''
        performs a training loop on self, with printed evaluations
        '''
        if fixed_seed:
            torch.manual_seed(fixed_seed)
        self.to(device)
        self.train()
        optimizer=torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-1)
        step=0
        report_on_batch_number=report_every_n_data//batch_size

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
                self.after_step_update(train_dataset)
        else:
            self.print_evaluation(train_loss=loss, eval_dataset=eval_dataset, step_number="Omega")
        self.eval()

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
        normalized_decoder = F.normalize(self.decoder, p=2, dim=1) #normalize columns
        hidden_layer=self.activation_function((residual_stream - self.decoder_bias) @ normalized_encoder + self.encoder_bias)
        reconstructed_residual_stream=hidden_layer @ normalized_decoder + self.decoder_bias
        loss= self.loss_function(residual_stream, hidden_layer, reconstructed_residual_stream) if compute_loss else None
        if not torch.is_inference_mode_enabled():
            dead_features_on_this_batch=self.find_dead_features(hidden_layer)
            self.num_data_since_last_activation=torch.where(dead_features_on_this_batch, 
                                                            self.num_data_since_last_activation+len(residual_stream),
                                                              0)
        return loss, residual_stream, hidden_layer, reconstructed_residual_stream

    @abstractmethod
    def activation_function(encoder_output):
        pass

    @abstractmethod
    def loss_function(encoder_output):
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
        pass

    def sparsity_loss_function(self, hidden_layer):
        decoder_column_norms=self.decoder.norm(dim=1)
        return torch.mean(hidden_layer*decoder_column_norms)
    
    def ghost_loss(self,residual_streams, reconstructed_residual_streams):
        dead_features=self.num_data_since_last_activation>=1000
        errors=residual_streams-reconstructed_residual_streams
        error_magnitudes=errors.norm(dim=-1)
        error_weighted_residual_stream=((residual_streams*error_magnitudes.unsqueeze(1)).sum(dim=0))/error_magnitudes.sum()
        error_weighted_error_direction=((errors*error_magnitudes.unsqueeze(1)).sum(dim=0))/error_magnitudes.sum()
        raw_encoder_ghost_loss=-1*error_weighted_residual_stream@(self.encoder[:, torch.where(dead_features>0)[0]])
        # encoder_ghost_loss=torch.nn.functional.relu(raw_encoder_ghost_loss).mean()
        encoder_ghost_loss=torch.nn.functional.softplus(raw_encoder_ghost_loss).sum()
        raw_decoder_ghost_loss=-1*error_weighted_error_direction@(self.decoder[torch.where(dead_features>0)[0]]).T
        # decoder_ghost_loss=torch.nn.functional.relu(raw_decoder_ghost_loss).mean()
        decoder_ghost_loss=torch.nn.functional.softplus(raw_decoder_ghost_loss).sum()
        # decoder_ghost_loss=0
        ghost_loss=(encoder_ghost_loss+decoder_ghost_loss)
        return ghost_loss

    def ghost_loss_anthropic(self, residual_streams, hidden_layers, reconstructed_residual_streams):
        residual=residual_streams-reconstructed_residual_streams
        with torch.no_grad():
            norm_of_reconstruction_residual=residual.norm(p=2, dim=1).mean()
        dead_features=self.find_dead_features(hidden_layers)
        ghost_hidden_layer=torch.exp(residual_streams@self.encoder+self.encoder_bias)[:,torch.where(dead_features)[0]]
        renormed_ghost_hidden_layers=1/2*norm_of_reconstruction_residual*(ghost_hidden_layer/ghost_hidden_layer.norm(dim=0))
        new_output=(renormed_ghost_hidden_layers@self.decoder[torch.where(dead_features)[0]])
        new_mean_square_error=(residual-new_output).norm(dim=1).mean()
        with torch.no_grad():
            scaling_factor=norm_of_reconstruction_residual/new_mean_square_error
        return new_mean_square_error*scaling_factor

    def reconstruction_error(self, residual_stream, reconstructed_residual_stream):
        reconstruction_l2=torch.norm(reconstructed_residual_stream-residual_stream, dim=-1)
        reconstruction_loss=(reconstruction_l2**2).mean()
        return reconstruction_loss
    
    def classify(self, dataset:PointCloudDataset):
        '''
            runs the dataset through this model, and returns classes for it based on the feature number
        '''
        losses, residual_streams, hidden_layers, reconstructed_residual_streams=self.catenate_outputs_on_dataset(dataset, include_loss=True)
        cluster_assignments=hidden_layers.argmax(dim=1)
        return cluster_assignments

class SAEAnthropic(SAETemplate):

    def __init__(self, anchor_points:torch.tensor, num_features:int, l1_sparsity_coefficient:float, ghost_loss_coefficient:float=.1, decoder_initialization_scale=0.1, cloud_scale_factor=1, ghost_loss_style="new"):
        super().__init__(anchor_points=anchor_points, 
                         num_features=num_features, 
                         decoder_initialization_scale=decoder_initialization_scale, 
                         cloud_scale_factor=cloud_scale_factor, 
                         ghost_loss_style=ghost_loss_style,
                         l1_sparsity_coefficient=l1_sparsity_coefficient,
                         ghost_loss_coefficient=ghost_loss_coefficient
                         )
    # def forward(self, embedded_points, compute_loss=False):
    #     hidden_layer=self.activation_function(embedded_points @ self.encoder + self.encoder_bias)
    #     reconstructed_embedded_points=hidden_layer @ self.decoder + self.decoder_bias
    #     if compute_loss:
    #         loss = self.loss_function(embedded_points, hidden_layer, reconstructed_embedded_points)
    #     else:
    #         loss = None
    #     self.update_
    #     return loss, embedded_points, hidden_layer, reconstructed_embedded_points
    
    def loss_function(self, residual_stream, hidden_layer, reconstructed_residual_stream):
        reconstruction_loss=self.reconstruction_error(residual_stream, reconstructed_residual_stream)
        sparsity_loss= self.sparsity_loss_function(hidden_layer)*self.l1_sparsity_coefficient
        if self.ghost_loss_style=="new":
            ghost_loss=self.ghost_loss(residual_stream, reconstructed_residual_stream)*self.ghost_loss_coefficient
        elif self.ghost_loss_style=="anthropic":
            ghost_loss=self.ghost_loss_anthropic(residual_stream, hidden_layer, reconstructed_residual_stream)
        total_loss=reconstruction_loss+sparsity_loss+ghost_loss
        return total_loss

    def activation_function(self, encoder_output):
        return F.relu(encoder_output)

    def report_model_specific_features(self):
        return [f"Sparsity loss coefficient: {self.l1_sparsity_coefficient}"]



class TopkSAE(SAETemplate):
    def __init__(self, anchor_points:torch.tensor, num_features: int, k:int, ghost_loss_coefficient:float=.1, decoder_initialization_scale=0.1, use_relu=True, l1_sparsity_coefficient=0, cloud_scale_factor=1, ghost_loss_style="new"):
        super().__init__(anchor_points=anchor_points, 
                         num_features=num_features, 
                         decoder_initialization_scale=decoder_initialization_scale, 
                         cloud_scale_factor=cloud_scale_factor, 
                         ghost_loss_style=ghost_loss_style,
                         ghost_loss_coefficient=ghost_loss_coefficient,
                         l1_sparsity_coefficient=l1_sparsity_coefficient)
        self.k=k
        self.use_relu=use_relu

    def activation_function(self, encoder_output):
        if self.use_relu:
            activations = torch.relu(encoder_output)
        else:
            activations=encoder_output
        with torch.no_grad():
            highest_indices=activations.argmax(dim=1)
            activations_mask=torch.zeros(activations.shape).scatter_(dim=1,index=highest_indices.unsqueeze(1), value=1)
        return activations*activations_mask

    # def forward(self, residual_stream, compute_loss=False):
    #     '''
    #     takes the trimmed residual stream of a language model (as produced by run_gpt_and_trim) and runs the SAE
    #     must return a tuple (loss, residual_stream, hidden_layer, reconstructed_residual_stream)
    #     residual_stream is shape (B, W, D), where B is batch size, W is (trimmed) window length, and D is the dimension of the model:
    #         - residual_stream is unchanged, of size (B, W, D)
    #         - hidden_layer is of shape (B, W, D') where D' is the size of the hidden layer
    #         - reconstructed_residual_stream is shape (B, W, D) 
    #     '''
    #     normalized_encoder = F.normalize(self.encoder, p=2, dim=1) #normalize columns
    #     # normalized_decoder = normalized_encoder.T
    #     normalized_decoder = F.normalize(self.decoder, p=2, dim=1) #normalize columns
    #     hidden_layer=self.activation_function((residual_stream - self.decoder_bias) @ normalized_encoder + self.encoder_bias)
    #     reconstructed_residual_stream=hidden_layer @ normalized_decoder + self.decoder_bias
    #     loss= self.loss_function(residual_stream, hidden_layer, reconstructed_residual_stream) if compute_loss else None
    #     return loss, residual_stream, hidden_layer, reconstructed_residual_stream

    def report_model_specific_features(self):
        return [f"k (sparsity): {self.k}", f"Leakiness: {self.leakiness}"]

    def loss_function(self, residual_stream, hidden_layer, reconstructed_residual_stream):
        reconstruction_loss=self.reconstruction_error(residual_stream, reconstructed_residual_stream)
        sparsity_loss= self.sparsity_loss_function(hidden_layer)*self.l1_sparsity_coefficient
        if self.ghost_loss_style=="new":
            ghost_loss=self.ghost_loss(residual_stream, reconstructed_residual_stream)*self.ghost_loss_coefficient
        elif self.ghost_loss_style=="anthropic":
            ghost_loss=self.ghost_loss_anthropic(residual_stream, hidden_layer, reconstructed_residual_stream)
        total_loss=reconstruction_loss+sparsity_loss+ghost_loss
        return total_loss


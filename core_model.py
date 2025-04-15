from dataclasses import dataclass, replace
from typing import List
import tensorflow as tf
import sonnet as snt

import functools

@dataclass
class EdgeSet:
    name: str
    features: tf.Tensor
    senders: tf.Tensor
    receivers: tf.Tensor


@dataclass
class MultiGraph:
    node_features: tf.Tensor
    edge_sets: List[EdgeSet]
    
class EncodeProcessDecode(snt.Module):
    def __init__(self,
                 output_size: int,
                 latent_size: int,
                 num_layers: int,
                 message_passing_steps: int,
                 message_passing_aggregator: str = 'sum',
                 attention: bool = False,
                 name: str = "EncodeProcessDecode"):
        super().__init__(name=name)

        self._latent_size = latent_size
        self._output_size = output_size
        self._num_layers = num_layers
        self._message_passing_steps = message_passing_steps
        self._message_passing_aggregator = message_passing_aggregator
        self._attention = attention
        
        def make_mlp(output_size, layer_norm=True):
            """
            Creates an MLP with optional layer normalization.

            Args:
                output_size (int): Final output dimensionality.
                layer_norm (bool): Whether to apply LayerNorm after the MLP.

            Returns:
                snt.Module: A sequential MLP model.
            """
            widths = [latent_size] * num_layers + [output_size]
            net = snt.nets.MLP(widths, activate_final=False)
            if layer_norm:
                return snt.Sequential([
                    net,
                    snt.LayerNorm(axis=-1, create_scale=True, create_offset=True)
                ])
            return net

        # Submodules
        self.encoder = Encoder(make_mlp=make_mlp, latent_size=latent_size)
        self.processor = ProcessorBlock(make_mlp=make_mlp,
                                        output_size=latent_size,
                                        message_passing_steps=message_passing_steps,
                                        aggregator=message_passing_aggregator,
                                        attention=attention)
        self.decoder = Decoder(make_mlp=make_mlp, output_size=output_size)

    
    def __call__(self, graph):
        """
        Full forward pass of Encode → Process → Decode.

        Args:
            graph (MultiGraph): Input graph with raw features.

        Returns:
            tf.Tensor: Decoded node-level outputs.
        """
        # Encode raw features into latent space
        latent_graph = self.encoder(graph)

        # Apply message passing through ProcessorBlock
        latent_graph = self.processor(latent_graph)

        # Decode node features into output
        output = self.decoder(latent_graph)

        return output



class Encoder(snt.Module):
    def __init__(self, make_mlp, latent_size, name="Encoder"):
        super().__init__(name=name)
        self._make_mlp = make_mlp
        self._latent_size = latent_size

        # MLPs for nodes and edges
        self.node_model = self._make_mlp(self._latent_size)
        self.edge_models = {}  # One per edge type

    def __call__(self, graph):
        """
        Args:
            graph (MultiGraph): Contains node_features and a list of edge_sets

        Returns:
            MultiGraph: With encoded node features and edge features
        """
        # Encode nodes
        node_latents = self.node_model(graph.node_features)

        # Encode edges, one model per edge type
        new_edge_sets = []
        for edge_set in graph.edge_sets:
            if edge_set.name not in self.edge_models:
                # Create and cache a model for this edge type
                self.edge_models[edge_set.name] = self._make_mlp(self._latent_size)
            edge_model = self.edge_models[edge_set.name]

            encoded_features = edge_model(edge_set.features)
            new_edge_set = replace(edge_set, features=encoded_features)
            new_edge_sets.append(new_edge_set)

        return MultiGraph(node_latents, new_edge_sets)


class Decoder(snt.Module):
    def __init__(self, make_mlp, output_size, name="Decoder"):
        super().__init__(name=name)
        self.model = make_mlp(output_size, layer_norm=False)

    def __call__(self, graph):
        """
        Args:
            graph (MultiGraph): A graph with encoded node features

        Returns:
            tf.Tensor: Decoded node-level outputs [num_nodes, output_size]
        """
        return self.model(graph.node_features)


class GraphNetBlock(snt.Module):
    def __init__(self, make_mlp, output_size, name="GraphNetBlock"):
        super().__init__(name=name)
        self._make_mlp = make_mlp
        self._output_size = output_size

        # One edge MLP per edge type
        self.edge_models = {}

        # Shared node MLP
        self.node_model = self._make_mlp(self._output_size)

    def _update_edge_features(self, node_features, edge_sets):
        """
        Updates all edge features based on node features.

        Args:
            node_features: [num_nodes, latent_dim]
            edge_sets: list of EdgeSet

        Returns:
            List of EdgeSet with updated features.
        """
        new_edge_sets = []

        for edge_set in edge_sets:
            senders = edge_set.senders
            receivers = edge_set.receivers

            sender_feats = tf.gather(node_features, senders)
            receiver_feats = tf.gather(node_features, receivers)

            concat_feats = tf.concat([sender_feats, receiver_feats, edge_set.features], axis=-1)

            # Build or retrieve the MLP for this edge type
            if edge_set.name not in self.edge_models:
                self.edge_models[edge_set.name] = self._make_mlp(self._output_size)
            edge_model = self.edge_models[edge_set.name]

            updated_edge_features = edge_model(concat_feats)

            new_edge_set = replace(edge_set, features=updated_edge_features)
            new_edge_sets.append(new_edge_set)

        return new_edge_sets

    def _update_node_features(self, node_features, edge_sets):
        """
        Updates node features by aggregating edge messages.

        Args:
            node_features: [num_nodes, latent_dim]
            edge_sets: list of EdgeSet with updated features

        Returns:
            tf.Tensor: Updated node features [num_nodes, latent_dim]
        """
        num_nodes = tf.shape(node_features)[0]
        aggregated_messages = []

        for edge_set in edge_sets:
            # Aggregate edge features for each receiver node
            messages = tf.math.unsorted_segment_sum(
                data=edge_set.features,
                segment_ids=edge_set.receivers,
                num_segments=num_nodes
            )
            aggregated_messages.append(messages)

        # Concatenate all message sources with the current node features
        all_messages = tf.concat([node_features] + aggregated_messages, axis=-1)

        # Update node features
        return self.node_model(all_messages)

    def __call__(self, graph):
        """
        Args:
            graph (MultiGraph): Contains node features and edge sets

        Returns:
            MultiGraph: Graph with updated node and edge features
        """
        # Save original features for residual connections
        original_node_features = graph.node_features
        original_edge_sets = graph.edge_sets

        # Step 1: Update edge features
        updated_edge_sets = self._update_edge_features(graph.node_features, graph.edge_sets)

        # Step 2: Update node features
        updated_node_features = self._update_node_features(graph.node_features, updated_edge_sets)

        # Step 3: Residual connections
        updated_node_features += original_node_features
        residual_edge_sets = [
            replace(es, features=es.features + orig_es.features)
            for es, orig_es in zip(updated_edge_sets, original_edge_sets)
        ]

        return MultiGraph(updated_node_features, residual_edge_sets)

class ProcessorBlock(snt.Module):
    def __init__(self,
                 make_mlp,
                 output_size,
                 message_passing_steps,
                 aggregator: str = 'sum',
                 attention: bool = False,
                 name="ProcessorBlock"):
        super().__init__(name=name)

        self.aggregator = aggregator  # for future use
        self.attention = attention 
        
        self.blocks = []
        for _ in range(message_passing_steps):
            block = GraphNetBlock(make_mlp=make_mlp, output_size=output_size)
            self.blocks.append(block)

    def __call__(self, graph):
        """
        Applies multiple message-passing steps.

        Args:
            graph (MultiGraph): Encoded graph from the encoder

        Returns:
            MultiGraph: Latent graph after message passing
        """
        for block in self.blocks:
            graph = block(graph)
        return graph


# if __name__ == '__main__':
#     # Dummy inputs
#     num_nodes = 4
#     num_edges = 6
#     node_dim = 3
#     edge_dim = 2

#     graph = MultiGraph(
#         node_features=tf.random.normal([num_nodes, node_dim]),
#         edge_sets=[
#             EdgeSet(
#                 name="mesh_edges",
#                 features=tf.random.normal([num_edges, edge_dim]),
#                 senders=tf.random.uniform([num_edges], maxval=num_nodes, dtype=tf.int32),
#                 receivers=tf.random.uniform([num_edges], maxval=num_nodes, dtype=tf.int32),
#             )
#         ]
#     )

#     # Create model
#     model = EncodeProcessDecode(
#         output_size=3,
#         latent_size=16,
#         num_layers=2,
#         message_passing_steps=3
#     )

#     # Run forward pass
#     output = model(graph)
#     print("Output shape:", output.shape)

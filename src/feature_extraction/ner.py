from src.feature_extraction.base import FeatureExtractor
import torch


class NERFeatureExtractorBasic(FeatureExtractor):

    def __init__(self, tokens_offset=(0,0)):
        self.tokens_offset = tokens_offset
        super().__init__(self.attention_to_feature)
    
    def attention_to_feature(self, att, ann, flatten=False, lengths=None, return_batch_ids=False, batch_id_offset=0):

        # attention: Tuple of tf.Tensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length)
        n_layers = len(att)
        batch_size, num_heads, sequence_length, sequence_length = att[0].size()

        features_batch = []
        labels_batch = []
        batch_ids = []
        for batch_i, annotations_in_sample in enumerate(ann):

            features_sample = []
            labels_sample = []
            for annotation in annotations_in_sample:
                batch_ids.append(batch_i + batch_id_offset)

                att_feature = torch.ones((n_layers, num_heads))
                
                for layer in range(n_layers):
                    # get attention from last token of span to first token of span
                    att_feature[layer, :] = att[layer][batch_i, :, annotation[0][1], annotation[0][0]]
                    
                features_sample.append(att_feature)
                labels_sample.append(annotation[1])

            features_batch.append(features_sample)
            labels_batch.append(labels_sample)
        
        if flatten:
            features_flat, labels_flat = [], []

            for ft_sample, lb_sample in zip(features_batch, labels_batch):
                
                features_flat.extend(ft_sample)
                labels_flat.extend(lb_sample)
            
            features_batch = torch.stack(features_flat)
            labels_batch = labels_flat

        if return_batch_ids:
            return features_batch, labels_batch, batch_ids

        return features_batch, labels_batch


class NERFeatureExtractorDiv(FeatureExtractor):

    def __init__(self, tokens_offset=(0,0)):
        self.tokens_offset = tokens_offset
        super().__init__(self.attention_to_feature)
    
    def attention_to_feature(self, att, ann, flatten=False, lengths=None, return_batch_ids=False, batch_id_offset=0):

        # attention: Tuple of tf.Tensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length)
        n_layers = len(att)
        batch_size, num_heads, sequence_length, sequence_length = att[0].size()

        features_batch = []
        labels_batch = []
        batch_ids = []
        for batch_i, annotations_in_sample in enumerate(ann):

            features_sample = []
            labels_sample = []
            for annotation in annotations_in_sample:
                batch_ids.append(batch_i + batch_id_offset)
                att_feature = torch.zeros((n_layers, num_heads))
                for layer in range(n_layers):
                    # get attention from last token of span to first token of span
                    att_feature[layer, :] = att[layer][batch_i, :, annotation[0][1], annotation[0][0]] / annotation[0][1]
                features_sample.append(att_feature)
                labels_sample.append(annotation[1])

            features_batch.append(features_sample)
            labels_batch.append(labels_sample)
        
        if flatten:
            features_flat, labels_flat = [], []

            for ft_sample, lb_sample in zip(features_batch, labels_batch):
                
                features_flat.extend(ft_sample)
                labels_flat.extend(lb_sample)
            
            features_batch = torch.stack(features_flat)
            labels_batch = labels_flat

        if return_batch_ids:
            return features_batch, labels_batch, batch_ids

        return features_batch, labels_batch
    


class NERFeatureExtractorThrowaway(FeatureExtractor):

    def __init__(self, tokens_offset=(0,0)):
        self.tokens_offset = tokens_offset
        super().__init__(self.attention_to_feature)
    
    def attention_to_feature(self, att, ann, flatten=False, lengths=None, return_batch_ids=False, batch_id_offset=0):

        # attention: Tuple of tf.Tensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length)
        n_layers = len(att)
        batch_size, num_heads, sequence_length, sequence_length = att[0].size()

        features_batch = []
        labels_batch = []
        batch_ids = []
        for batch_i, annotations_in_sample in enumerate(ann):

            features_sample = []
            labels_sample = []
            for annotation in annotations_in_sample:
                batch_ids.append(batch_i + batch_id_offset)
                att_feature = torch.ones((n_layers, num_heads*4)) * -torch.inf
                #att_feature = torch.ones((n_layers, num_heads))
                att_feature[:, 3*num_heads:] = torch.inf
                
                for layer in range(n_layers):
                    # get attention from last token of span to first token of span
                    att_feature[layer, :num_heads] = att[layer][batch_i, :, annotation[0][1], annotation[0][0]]
                    # B token
                    att_feature[layer, num_heads:2*num_heads] = att[layer][batch_i, :, annotation[0][0], annotation[0][0]-1]
                    for i in range(annotation[0][0], annotation[0][1]+1):
                        """
                        print("------------------------------------------------")
                        print(torch.max(torch.stack([att[layer][batch_i, :, i, i-1], att_feature[layer, num_heads:2*num_heads]]), dim=0)[0].size())
                        print(att[layer][batch_i, :, i, i-1], att_feature[layer, num_heads:2*num_heads])
                        print(torch.max(torch.stack([att[layer][batch_i, :, i, i-1], att_feature[layer, num_heads:2*num_heads]]), dim=0)[0])
                        print(att[layer][batch_i, :, i, i-1], att_feature[layer, 2*num_heads:])
                        print(torch.min(torch.stack([att[layer][batch_i, :, i, i-1], att_feature[layer, 2*num_heads:]]), dim=0)[0])
                        """
                        att_feature[layer, 2*num_heads:3*num_heads] = torch.max(torch.stack([att[layer][batch_i, :, i, i-1], att_feature[layer, 2*num_heads:3*num_heads]]), dim=0)[0]
                        att_feature[layer, 3*num_heads:] = torch.min(torch.stack([att[layer][batch_i, :, i, i-1], att_feature[layer, 3*num_heads:]]), dim=0)[0]

                    # att_feature[layer, :] = torch.mean(att[layer][batch_i, :, lengths[batch_i]-1, annotation[0][0]:annotation[0][1]+1],dim=-1)

                features_sample.append(att_feature)
                labels_sample.append(annotation[1])

            features_batch.append(features_sample)
            labels_batch.append(labels_sample)
        
        if flatten:
            features_flat, labels_flat = [], []

            for ft_sample, lb_sample in zip(features_batch, labels_batch):
                
                features_flat.extend(ft_sample)
                labels_flat.extend(lb_sample)
            
            features_batch = torch.stack(features_flat)
            labels_batch = labels_flat

        if return_batch_ids:
            return features_batch, labels_batch, batch_ids

        return features_batch, labels_batch

class NERFeatureExtractorMLP(FeatureExtractor):

    def __init__(self, tokens_offset=(0,0)):
        self.tokens_offset = tokens_offset
        super().__init__(self.attention_to_feature)
    
    def attention_to_feature(self, att, mlp, ann, flatten=False, lengths=None, return_batch_ids=False, batch_id_offset=0):

        # attention: Tuple of tf.Tensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length)
        n_layers = len(att)
        batch_size, num_heads, sequence_length, sequence_length = att[0].size()

        features_batch = []
        labels_batch = []
        batch_ids = []
        for batch_i, annotations_in_sample in enumerate(ann):

            features_sample = []
            labels_sample = []
            for annotation in annotations_in_sample:
                batch_ids.append(batch_i + batch_id_offset)

                att_feature = torch.ones((n_layers, num_heads))
                
                for layer in range(n_layers):
                    # get attention from last token of span to first token of span
                    att_feature[layer, :] = att[layer][batch_i, :, annotation[0][1], annotation[0][0]]

                att_feature = torch.cat((att_feature.view(-1, 1), mlp[3*len(mlp)//4][batch_i,annotation[0][1]].unsqueeze(-1)))

                features_sample.append(att_feature)
                labels_sample.append(annotation[1])

            features_batch.append(features_sample)
            labels_batch.append(labels_sample)
        
        if flatten:
            features_flat, labels_flat = [], []

            for ft_sample, lb_sample in zip(features_batch, labels_batch):
                
                features_flat.extend(ft_sample)
                labels_flat.extend(lb_sample)
            
            features_batch = torch.stack(features_flat)
            labels_batch = labels_flat

        if return_batch_ids:
            return features_batch, labels_batch, batch_ids

        return features_batch, labels_batch

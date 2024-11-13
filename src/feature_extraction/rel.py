

from src.feature_extraction.base import FeatureExtractor
from src.feature_extraction.ner import NERFeatureExtractorBasic
import torch


class SpanPairFeatureExtractor(FeatureExtractor):

    def __init__(self, mode="first2last", pooling="mean"):
        self.mode = mode

        if mode not in ["first2last", "last2last", "pooled2last"]:
            raise NotImplementedError()
        
        if mode in ["first2last", "last2last", "pooled2last"]:
            # since we are not doing any pooling we can just use the same extractor as for mention detection
            self.subextractor = NERFeatureExtractorBasic()

        self._pooling = pooling

        super().__init__(self.attention_to_feature)
    
    def pooling(self, x):
        if self._pooling == "mean":
            return torch.mean(x, dim=-1)
        elif self._pooling == "max":
            return torch.max(x, dim=-1)[0]

    def attention_to_feature(self, att, ann, flatten=False, lengths=None, return_batch_ids=False, batch_id_offset=0):

        if self.mode == "first2last":
            # feature is attention from first token of span_b to last token of span_a
            new_anns = []
            for batch_i in ann:
                anns_i = [((x[0][0][1], x[0][1][0]), x[1]) for x in batch_i]
                new_anns.append(anns_i)
            return self.subextractor(att, new_anns, flatten=flatten, lengths=lengths, return_batch_ids=return_batch_ids, batch_id_offset=batch_id_offset)

        if self.mode == "last2last":
            # feature is attention from last token of span_b to last token of span_a
            new_anns = []
            for batch_i in ann:
                anns_i = [((x[0][0][1], x[0][1][1]), x[1]) for x in batch_i]
                new_anns.append(anns_i)
            return self.subextractor(att, new_anns, flatten=flatten, lengths=lengths, return_batch_ids=return_batch_ids, batch_id_offset=batch_id_offset)

        if self.mode == "pooled2last":
            # feature is pooled attention from each token in span_b to last token of span_a
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
                        span_attentions = att[layer][batch_i, :, annotation[0][1][0]:annotation[0][1][1]+1, annotation[0][0][1]].detach().cpu()
                        att_feature[layer, :] = self.pooling(span_attentions)

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


class RelationFeatureExtractor(FeatureExtractor):

    def __init__(self, **kwargs):

        self.subextractor = SpanPairFeatureExtractor(**kwargs)

        super().__init__(self.attention_to_feature)

    def attention_to_feature(self, att, ann, entity_clusters, flatten=False, lengths=None, return_ann_positions=False, return_batch_ids=False, batch_id_offset=0):

        expanded_anns_batch = []
        ann_positions = []
        for batch_i, annotations_in_sample in enumerate(ann):

            expanded_anns = []
            for batch_j, annotation in enumerate(annotations_in_sample):
                
                head_entity_cluster_id = annotation[0][0]
                tail_entity_cluster_id = annotation[0][1]

                # fetch mentions
                head_entity_mentions = entity_clusters[batch_i][head_entity_cluster_id]
                tail_entity_mentions = entity_clusters[batch_i][tail_entity_cluster_id]

                for head_entity in head_entity_mentions:
                    for tail_entity in tail_entity_mentions:
                        
                        if head_entity[0][1] < tail_entity[0][0]:
                            # direction h -> t
                            expanded_anns.append(((head_entity[0], tail_entity[0]), annotation[1]))
                        else:
                            # direction t -> h
                            """
                            if annotation[1] == "NONE":
                                new_label = annotation[1]
                            else:
                                new_label = annotation[1] + "_REV"
                            """
                            expanded_anns.append(((head_entity[0], tail_entity[0]), annotation[1]))
                        ann_positions.append((batch_i, batch_j))
            expanded_anns_batch.append(expanded_anns)

        if flatten:
            if return_ann_positions:
                    return self.subextractor(att, expanded_anns_batch, flatten=flatten, lengths=lengths, return_batch_ids=return_batch_ids, batch_id_offset=batch_id_offset), ann_positions
            return self.subextractor(att, expanded_anns_batch, flatten=flatten, lengths=lengths, return_batch_ids=return_batch_ids, batch_id_offset=batch_id_offset)
        
        # rebuild correct structure
        flat_anns, flat_labels = self.subextractor(att, expanded_anns_batch, flatten=True, lengths=lengths, return_batch_ids=return_batch_ids, batch_id_offset=batch_id_offset)

        output_ft = [[[] for _ in x] for x in ann]
        output_lb = [[[] for _ in x] for x in ann]

        for f, l, idx in zip(flat_anns, flat_labels, ann_positions):
            output_ft[idx[0]][idx[1]].append(f)
            output_lb[idx[0]][idx[1]].append(l)

        return output_ft, output_lb



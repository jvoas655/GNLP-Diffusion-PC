import torch
from torch import nn

from math import floor


class ContrastiveLoss(nn.Module):
    """
    Inspiration for this loss comes from SIMCLR, an implementation with details can be found here
    https://zablo.net/blog/post/understanding-implementing-simclr-guide-eli5-pytorch/
    although it was originally intended for images, the embedding contrastive algorithm works anywhere.
    """

    negative_sample: float
    negative_temperature: float

    def __init__(
            self,
            negative_temperature: float = 0.5
    ):
        """
        :param negative_temperature: <float> A trade off hyperparameter between the weight of positive examples and negative
            examples.  A higher value will put more weight on negative examples.
        """
        super().__init__()

        self.negative_temperature = negative_temperature

        self.__negative_weight__ = 1 - self.negative_temperature if self.negative_temperature < 1 else 0.001
        self.__positive_weight__ = self.negative_temperature if self.negative_temperature > 0 else 0.001

        self.simularity = nn.CosineSimilarity(dim=0)

    def forward(self, embeddings: torch.Tensor, targets: torch.Tensor):
        assert embeddings.shape[0] == targets.shape[0]

        batch_size = embeddings.shape[0]

        # To avoid exploding gradients, we want to allow the embeddings to be of any size but for loss we will restrict
        # the embeddings to the unit sphere.
        normalized_embeddings = nn.functional.normalize(embeddings, dim=1)
        normalized_targets = nn.functional.normalize(targets, dim=1)

        # Get the positive simularity measures for each embedding and their targets.
        positive_simularities = nn.functional.cosine_similarity(normalized_embeddings, normalized_targets)
        negative_simularities = nn.functional.cosine_similarity(
            normalized_embeddings.unsqueeze(1),
            normalized_embeddings.unsqueeze(0),
            dim=2
        )

        # Get the loss for each type of embedding (embedding to target, embedding to other embedding)
        positive_ex_loss = torch.exp(positive_simularities / self.__positive_weight__)
        negative_ex_loss = torch.exp(negative_simularities / self.__negative_weight__)

        # This is the actual loss per embedding
        loss_partial = -torch.log(positive_ex_loss / torch.sum(negative_ex_loss, dim=1))

        # Average loss over the entire batch.
        # 2 * batch size because we are effectively counting each embedding twice.
        # i -> j embedding and j -> i embedding.  You could cut the matrix in half and do the same
        # loss, but it's just a scalar so unless computation becomes heavy there's no need.
        loss = torch.sum(loss_partial) / (2 * batch_size)
        return loss



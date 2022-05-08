from argparse import ArgumentParser
from pathlib import Path
from typing import List
import h5py
from rouge_score import rouge_scorer
from tqdm import tqdm
import matplotlib.pyplot as plt
from random import choices
from sklearn.decomposition import PCA
import plotly.graph_objs as go
import numpy as np

import torch.nn.functional as f
from src.utilities.paths import DATA_FOLDER
from src.gnlp_diffusion_point_cloud.models.backbones.t5 import T5Backbone, T5Size
from gnlp_diffusion_point_cloud.utils.dataset import cate_to_synsetid, synsetid_to_cate


def rouge_simularity_matrix(
        descriptions: List[str],
        rouge_score_type: str = 'rouge1',
        progress_bar: bool = False
) -> List[List[float]]:

    scorer = rouge_scorer.RougeScorer([rouge_score_type], use_stemmer=True)
    sim_matrix = []

    for idx, description in tqdm(
            enumerate(descriptions),
            total=len(descriptions),
            desc='Building simularity matrix',
            disable=not progress_bar,
            position=0,
            leave=False
    ):
        sims = [x[idx] for x in sim_matrix]
        for oidx, other_description in tqdm(
                enumerate(descriptions[idx:]),
                total=len(descriptions) - idx,
                desc='Comparing description with others',
                disable=not progress_bar,
                position=1,
                leave=False
        ):
            scores = scorer.score(description, other_description)
            sims.append(scores[rouge_score_type].fmeasure)
        sim_matrix.append(sims)

    return sim_matrix


def t5_embedding(
        descriptions: List[str],
        t5_size: T5Size = T5Size.small,
        progress_bar: bool = False
):
    encoder = T5Backbone(size=t5_size)

    embeddings = []
    for idx, description in tqdm(
        enumerate(descriptions),
        total=len(descriptions),
        desc='Embedding descriptions',
        disable=not progress_bar,
        position=0,
        leave=False
    ):
        embedding = encoder.encode(encoder.tokenizer(description))
        embedding = embedding.view(encoder.token_length, -1)
        embeddings.append(f.max_pool1d(embedding, kernel_size=embedding.shape[-1]).squeeze().detach().numpy())

    return np.stack(embeddings)


def plot_rouge_sim(descriptions: List[str], output: Path = None):
    sim_matrix = rouge_simularity_matrix(descriptions, progress_bar=True)

    fig, ax = plt.subplots(figsize=(20,20))
    cax = ax.matshow(sim_matrix, interpolation='nearest')
    ax.grid(True)
    plt.title('Rouge Simularity Matrix')

    fig.colorbar(cax, ticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, .75, .8, .85, .90, .95, 1])
    plt.show()
    plt.savefig(str(output))


def plot_t5_pca(descriptions: List[str], topn: int = -1, output: Path = None):
    embeddings = t5_embedding(descriptions)

    if topn == -1:
        topn = len(embeddings)

    three_dim = PCA(random_state=0).fit_transform(embeddings)[:, :3]
    # For 2D, change the three_dim variable into something like two_dim like the following:
    # two_dim = PCA(random_state=0).fit_transform(word_vectors)[:,:2]

    data = []
    count = 0

    for i in range(len(embeddings)):
        trace = go.Scatter3d(
            x=three_dim[count:count + topn, 0],
            y=three_dim[count:count + topn, 1],
            z=three_dim[count:count + topn, 2],
            text=list(range(count, count + topn)),
            name=i,
            textposition="top center",
            textfont_size=20,
            mode='markers+text',
            marker={
                'size': 10,
                'opacity': 0.8,
                'color': 2
            }

        )

        # For 2D, instead of using go.Scatter3d, we need to use go.Scatter and delete the z variable. Also, instead of using
        # variable three_dim, use the variable that we have declared earlier (e.g two_dim)

        data.append(trace)
        count = count + topn

    trace_input = go.Scatter3d(
        x=three_dim[count:, 0],
        y=three_dim[count:, 1],
        z=three_dim[count:, 2],
        text=list(range(count, count+topn)),
        name='input words',
        textposition="top center",
        textfont_size=20,
        mode='markers+text',
        marker={
            'size': 10,
            'opacity': 1,
            'color': 'black'
        }
    )

    # For 2D, instead of using go.Scatter3d, we need to use go.Scatter and delete the z variable.  Also, instead of using
    # variable three_dim, use the variable that we have declared earlier (e.g two_dim)

    data.append(trace_input)

    # Configure the layout

    layout = go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=True,
        legend=dict(
            x=1,
            y=0.5,
            font=dict(
                family="Courier New",
                size=25,
                color="black"
            )),
        font=dict(
            family=" Courier New ",
            size=15),
        autosize=False,
        width=1000,
        height=1000
    )

    plot_figure = go.Figure(data=data, layout=layout)
    plot_figure.show()

    if output:
        filename = str(output)
        if not filename.endswith('.html'):
            filename += '.html'
        plot_figure.write_html(filename)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--input', '-i', type=str, required=True,
                           help="Path to the dataset file containing descriptions of objects")
    parser.add_argument('--comparison_method', '-cm', type=str, required=True,
                           choices=['rouge', 'T5', 'CustomModel'],
                           help='The method to compare descriptions against each other with.  A custom model will '
                                'require a path to a .pth file specified by --custom_model_pth')
    parser.add_argument('--categories', '-c', type=str, nargs='+', default=(),
                        help='Categories you want to visualize (ex. table chair bus)')
    parser.add_argument('--split', '-s', type=str, choices=['train', 'val', 'test'], default='train',
                        help='Data split to view.  Options are train, val, and test.')
    parser.add_argument('--sample_size', '-ss', type=float, default=0.0002,
                        help='Percent of the data you want to visualize (1 would be all, but very slow)')
    parser.add_argument('--output_file', '-o', type=str, default=None,
                           help='Path to the output file that the evaluations will be stored in')
    parser.add_argument('--force_output', '-f', dest='force_output', action='store_true',
                           help='Overwrite anything currently written in the output file path.')

    args = parser.parse_args()

    input_file: Path = Path(args.input) if '/' in args.input else DATA_FOLDER / args.input
    output_file: Path = Path(args.output_file) if args.output_file else None
    force_output: bool = args.force_output
    categories: List[str] = args.categories
    split: str = args.split
    sample_size: float = args.sample_size
    comparison_method: str = args.comparison_method

    assert input_file.exists(), f'{input_file} does not exist!'
    assert not output_file or not output_file.exists() or force_output, \
        'Please specify an empty file path for the output parameter -o OR specify the force flag -f'

    data = h5py.File(str(input_file))

    if len(categories) == 0:
        categories = list(data.keys())
    else:
        categories = [cate_to_synsetid[x] for x in categories]

    descriptions_to_compare = []
    for category in categories:
        descriptions = data[category][split]
        descriptions_to_compare.extend([x.decode('utf-8') for x in descriptions])

    sampled_descriptions = choices(descriptions_to_compare, k=int(len(descriptions_to_compare) * sample_size))

    if comparison_method == 'T5':
        plot_t5_pca(sampled_descriptions, output=output_file)
    elif comparison_method == 'rouge':
        plot_rouge_sim(sampled_descriptions, output=output_file)

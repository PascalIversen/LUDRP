import pandas as pd
import numpy as np
from .GDSC_feature_extractor import cosmic_ids_to_cell_line_names


def get_gdsc_mutation_similarity(
    kernel="jaccard", path_mutations="data/GDSC/mutations_20191101.csv"
):
    """
    get gene mutation similiarity matrix of cell_lines or get the gene mutation feature matrix
    (if kernel=None)

    returns:
    pandas DataFrame: cell_line x cell_line (or if kernel= None: cell_line x genes)

    kernel:
    "jaccard": Jaccard similarity
    None: return cmutation features

    path_mutations:
    path to mutation data, get from: https://cellmodelpassports.sanger.ac.uk/downloads

    """
    mutations = pd.read_csv(path_mutations)

    # pivot to have a cells x genes table
    mutations = mutations.drop_duplicates(["model_name", "gene_symbol"]).pivot(
        index="model_name", columns="gene_symbol", values="cancer_driver"
    )
    mutations = mutations.fillna("not_mutated")
    mutations = mutations.replace(to_replace=True, value="mutated")
    mutations = mutations.replace(to_replace=False, value="mutated")
    mutations = mutations == "mutated"

    # drop rarely mutated genes
    n_mutations_in_all_cell_lines_per_gene = mutations.sum()
    genes_which_are_rarely_mutated = n_mutations_in_all_cell_lines_per_gene.index[
        n_mutations_in_all_cell_lines_per_gene < 5
    ]
    mutations = mutations.drop(genes_which_are_rarely_mutated, axis=1)

    mutations.index = mutations.index.astype(str)

    if kernel is None:
        return mutations

    elif kernel == "jaccard":
        similarity = pd.DataFrame(columns=mutations.index, index=mutations.index)

        from scipy.spatial.distance import jaccard  # intercept over union
        from tqdm._tqdm_notebook import tqdm_notebook

        for cell_from in tqdm_notebook(mutations.index):
            for cell_to in mutations.index:
                # TODO try higher weight for cancer driver genes in jaccard call
                if np.isnan(similarity[cell_from].loc[cell_to]):
                    similarity[cell_from].loc[cell_to] = 1 - jaccard(
                        mutations.loc[cell_from], mutations.loc[cell_to]
                    )
                    similarity[cell_to].loc[cell_from] = similarity[cell_from].loc[
                        cell_to
                    ]
        return similarity
    else:
        raise ValueError("Invalid kernel specified.")


def get_gdsc_copy_number_var_similarity(
    data_type="gistic", kernel="kendall", path_cnv=None
):
    """
    get copy number variation similiarity matrix of cell_lines or get the cell_line copy number variation feature matrix
    (if kernel=None)

    returns:
    pandas DataFrame: cell_line x cell_line (or if kernel= None: cell_line x genes)

    data_type:
    use absolute PICNIC or GISTIC data, get from: https://cellmodelpassports.sanger.ac.uk/downloads

    kernel:
    "kendall": Kendall’s tau correlation measure for ordinal data.

    None: return copy number variation feature matrix

    """
    from tqdm import tqdm_notebook

    if path_cnv is None:
        if data_type == "picnic":
            path_cnv = "data/GDSC/cnv_abs_copy_number_picnic_20191101.csv"
        elif data_type == "gistic":
            path_cnv = "data/GDSC/cnv_gistic_20191101.csv"
        else:
            raise ValueError("""data_type has to be "picnic" or "gistic" """)

    cnv_norm = pd.read_csv(path_cnv, header=[0, 1], index_col=[0, 1])
    cnv_norm.columns = cnv_norm.columns.droplevel(0)
    cnv_norm.index = cnv_norm.index.droplevel(0)
    cnv_norm.index = cnv_norm.index.astype(str)

    if kernel is None:
        return cnv_norm

    elif kernel == "kendall":
        from scipy.stats import kendalltau

        cell_lines = cnv_norm.columns
        similarity = pd.DataFrame(columns=cell_lines, index=cell_lines)

        for i, cell_line_1 in tqdm_notebook(
            enumerate(cell_lines), total=len(cell_lines)
        ):
            for cell_line_2 in cell_lines[i:]:
                kendall_tau = kendalltau(
                    cnv_norm.loc[:, cell_line_1], cnv_norm.loc[:, cell_line_2]
                )

                # symmetric matrix:
                similarity.loc[cell_line_1, cell_line_2] = kendall_tau[0]
                similarity.loc[cell_line_2, cell_line_1] = kendall_tau[0]

        # kendalls tau ranges from -1 to 1, scale to 0,1
        assert np.isclose(similarity.max().max(), 1.0)
        minimum = similarity.min().min()
        similarity = (similarity - minimum) / (1.0 - minimum)
        return similarity

    else:
        raise ValueError(
            """ Invalid kernel specified, choose "kendall" or None (not  "None") """
        )


def get_gdsc_methylation_similarity(
    kernel="rbf",
    rbf_gamma=0.005,
    path_methylation="data\GDSC\F2_METH_CELL_DATA.txt",
    path_methylation_annotations="data\GDSC\methSampleId_2_cosmicIds.xlsx",
):
    """
    get DNA methylation similiarity matrix of cell_lines or cell_line features (if kernel=None)

    returns:
    pandas DataFrame: cell_line_names x cell_line_names (or if kernel= None: cell_line_names x methylation sites)

    kernel:
    "pearson_corr": Pearson correlation coefficients of the profiles
    "rbf": Radial basis function similiarity of the profiles
    "manhattan": Manhattan distance of the profiles
    None: return features

    rbf_gamma :
    float, gamma parameter of the RBF

    path_methylation: path of methylation data
    get from: https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources/Home.html

    path_methylation_annotation: path of methylation annotations
    get from: https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources/Home.html

    """
    methylation = pd.read_csv(path_methylation, sep="\t", index_col=0)
    methylation_annotation = pd.read_excel(
        path_methylation_annotations, engine="openpyxl"
    )

    methylation_id_to_cell_line_name_map = {
        f"{sentrix}_{position}": cell_line_name
        for sentrix, position, cell_line_name in zip(
            methylation_annotation.Sentrix_ID,
            methylation_annotation.Sentrix_Position,
            methylation_annotation.Sample_Name,
        )
    }

    methylation.columns = [
        methylation_id_to_cell_line_name_map[c] for c in methylation.columns
    ]
    methylation = methylation.transpose()
    # remove duplicated cell line names
    methylation = methylation.loc[~methylation.index.duplicated(keep="first")]
    methylation.index = methylation.index.astype(str)

    if kernel == "pearson_corr":
        similiarity_matrix = np.corrcoef(x=methylation)
        similiarity_matrix = pd.DataFrame(
            similiarity_matrix, index=methylation.index, columns=methylation.index
        )

        return similiarity_matrix

    elif kernel == "rbf":
        from sklearn.metrics.pairwise import rbf_kernel

        similiarity_matrix = rbf_kernel(X=methylation, gamma=rbf_gamma)
        similiarity_matrix = pd.DataFrame(
            similiarity_matrix, index=methylation.index, columns=methylation.index
        )

        return similiarity_matrix
    elif kernel == "manhattan":
        from sklearn.metrics.pairwise import manhattan_distances

        similiarity_matrix = manhattan_distances(X=methylation)
        similiarity_matrix = 1 - similiarity_matrix / similiarity_matrix.max()
        similiarity_matrix = pd.DataFrame(
            similiarity_matrix, index=methylation.index, columns=methylation.index
        )

        return similiarity_matrix
    elif kernel == None:
        return methylation

    else:
        raise ValueError(
            """ Invalid kernel specified, choose "rbf" or "pearson_corr" or None """
        )


def get_gdsc_gene_expression_similarity(
    path_gene_expression="data/GDSC/Cell_line_RMA_proc_basalExp.txt",
    path_cell_annotations="data/GDSC/Cell_Lines_Details.csv",
    kernel="rbf",
    rbf_gamma=None,
    scaling=None,  # or minmax
):
    """
    get gene expression similiarity matrix of cell_lines or cell_line features (if kernel=None)

    returns:
    pandas DataFrame: cell_line_names x cell_line_names (or if kernel= None: cell_line_names x genes)

    if there is no gene expression data for a cell_line in cell_line_names, it is not part of the final DataFrame

    cell_line_names:
    list of str: GDSC cell-line names

    kernel:
    "pearson_corr": Pearson correlation coefficients of the gene expression profiles
    "rbf": Radial basis function similiarity of the gene expression profiles
    None: return features

    rbf_gamma :
    float, gamma parameter of the RBF, if None, defaults to 1.0 / n_features

    path_gene_expression: RMA normalised expression data for cell-lines
    get from: https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources/Home.html

    path_cell_annotations: path of GDSC cell-line annotations
    get from: https://www.cancerrxgene.org/downloads/bulk_download

    scaling: normalizer of the features, can be "minmax" or "None" for no scaling

    """

    # gene expression data
    try:
        gene_expression = pd.read_csv(path_gene_expression, sep="\t")
    except:
        print(
            "gene expression data import failed, maybe sep and decimal keywords need to be adapted or the path is wrong"
        )
        raise

    try:
        gene_expression = gene_expression.drop(["GENE_SYMBOLS", "GENE_title"], axis=1)
    except KeyError:
        print("Warning: structure of gene expression data changed")

    if scaling == "minmax":
        from sklearn.preprocessing import minmax_scale

        gene_expression = pd.DataFrame(
            minmax_scale(gene_expression, axis=1),
            columns=gene_expression.columns,
            index=gene_expression.index,
        )

    # refractor column names to cosmic id and then map to cell-line name
    ge_columns = [x[5:] for x in list(gene_expression.columns)]  # remove "DATA" prefix
    ge_columns = cosmic_ids_to_cell_line_names(ge_columns, path_cell_annotations)
    gene_expression.columns = ge_columns

    if kernel == "pearson_corr":
        import numpy as np

        similiarity_matrix = np.corrcoef(x=gene_expression.transpose())
        similiarity_matrix = pd.DataFrame(
            similiarity_matrix,
            index=gene_expression.columns,
            columns=gene_expression.columns,
        )

    elif kernel == "rbf":
        from sklearn.metrics.pairwise import rbf_kernel

        similiarity_matrix = rbf_kernel(X=gene_expression.transpose(), gamma=rbf_gamma)
        similiarity_matrix = pd.DataFrame(
            similiarity_matrix,
            index=gene_expression.columns,
            columns=gene_expression.columns,
        )

    elif kernel == None:
        return gene_expression.transpose()

    else:
        raise ValueError(
            """ Invalid kernel specified, choose "rbf" or "pearson_corr" or None """
        )

    return similiarity_matrix

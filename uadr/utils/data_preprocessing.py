from typing import List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .GDSC_feature_extractor import cosmic_ids_to_cell_line_names


def get_gdsc_gene_expression(
    genes: List[str],
    path_cell_annotations: str = "GDSC/Cell_Lines_Details.csv",
    path_gene_expression: str = "GDSC/Cell_line_RMA_proc_basalExp.txt",
    verbose: bool = True,
    data_dir="data"
):
    """
    Return the gene expression dataframe(n_cells x n_genes)
    for a set of gene symbols for all cell_lines of the GDSC cell line annotation file.
    """
    path_cell_annotations = f"{data_dir}/{path_cell_annotations}"
    path_gene_expression = f"{data_dir}/{path_gene_expression}"

    gene_expression = pd.read_csv(path_gene_expression, sep="\t")

    gene_expression = gene_expression.drop(["GENE_title"], axis=1).set_index(
        "GENE_SYMBOLS"
    )
    gene_expression.index = gene_expression.index.astype(str)

    # refactor column names to cosmic id and then map to cell-line name
    ge_columns = [
        x.split("DATA.")[1] for x in list(gene_expression.columns)
    ]  # remove "DATA" prefix
    ge_columns = cosmic_ids_to_cell_line_names(
        ge_columns, path_cell_annotations=path_cell_annotations
    )
    gene_expression.columns = ge_columns.astype(str)

    # filter out the genes
    genes = [g for g in genes if g in gene_expression.index]
    assert len(genes) > 0, "No gene expression data for the queried genes."

    if verbose:
        number_of_queried_genes = len(genes)
        print(
            f"no data for {number_of_queried_genes - len(genes)} of {number_of_queried_genes} queried genes.",
            flush=True,
        )

    gene_expression = gene_expression.loc[genes]

    return gene_expression.T


def get_cell_line_name_to_gene_expression_map(
    relevant_genes: List[str],
    padding: bool = False,
    return_gene_list: bool = False,
    verbose: bool = True,
    data_dir="data"
):
    """
    Return a dictionary mapping cell line names to gene expression vectors.
    If return_gene_list is True, also return the list of genes in the vectors.

    """
    gene_expression = get_gdsc_gene_expression(relevant_genes, verbose=verbose, data_dir=data_dir)
    gene_list = gene_expression.columns

    cell_line_name_to_gene_expression_map = {}
    for cell_line in gene_expression.index:
        cell_line_name_to_gene_expression_map[cell_line] = gene_expression.loc[
            cell_line
        ].values

    if padding:
        cell_line_name_to_gene_expression_map["padding"] = np.zeros(
            len(gene_expression.loc[cell_line].values)
        ).astype(float)

    if not return_gene_list:
        return cell_line_name_to_gene_expression_map
    else:
        return cell_line_name_to_gene_expression_map, gene_list


def get_drug_name_to_fingerprint_map(n_bits: int = 128, padding: bool = False, data_dir="../data"):
    drug_name_to_fingerprint_map = np.load(
        f"{data_dir}/GDSC/drug_fingerprints/drug_name_to_demorgan_{n_bits}_map.npy",
        allow_pickle=True,
    ).item()
    for drug in drug_name_to_fingerprint_map:
        drug_name_to_fingerprint_map[drug] = drug_name_to_fingerprint_map[drug].astype(
            float
        )
    if padding:
        drug_name_to_fingerprint_map["padding"] = np.zeros(n_bits).astype(float)
    return drug_name_to_fingerprint_map


def get_preprocessed_features(
    current_split: str,
    base_path: str,
    cross_validation_type: str,
    scaling_mode: str,
    n_bits_drug_fingerprints: int,
    data_dir="data"
):
    path = (
        f"{base_path}splits/response_matrices/"
        f"drug_response_matrices_{cross_validation_type}_{scaling_mode}_fold{current_split}.npy"
    )
    drug_response_matrices = np.load(path, allow_pickle=True).item()

    cell_lines = drug_response_matrices["train"].index
    drugs = drug_response_matrices["train"].columns
    relevant_genes = list(
        pd.read_csv(
            f"{data_dir}/gene_list_paccmann_network_prop.txt", header=None
        ).values.flatten()
    )

    (
        cell_line_name_to_gene_expression_map,
        gene_list,
    ) = get_cell_line_name_to_gene_expression_map(
        relevant_genes=relevant_genes, return_gene_list=True, verbose=False, data_dir=data_dir
    )
    cell_line_features = []
    for cl in cell_lines:
        if cl in cell_line_name_to_gene_expression_map:
            cell_line_features.append(cell_line_name_to_gene_expression_map[cl])
        else:
            cell_line_features.append(np.nan)
    has_data = ~np.array([np.any(np.isnan(clf)) for clf in cell_line_features])
    cell_line_features = np.stack(
        np.array(cell_line_features, dtype=object)[has_data]
    ).astype(float)
    cell_lines = cell_lines[has_data]

    # arcsinh scaling for the cell line features
    cell_line_features = np.arcsinh(cell_line_features)

    for i, cl in enumerate(cell_lines):
        cell_line_name_to_gene_expression_map[cl] = cell_line_features[i, :]

    # get drug features
    drug_name_to_fingerprint_map = get_drug_name_to_fingerprint_map(
        n_bits=n_bits_drug_fingerprints, data_dir=data_dir
    )
    drug_features = []
    for d in drugs:
        if d in drug_name_to_fingerprint_map:
            drug_features.append(drug_name_to_fingerprint_map[d])
        else:
            drug_features.append(np.nan)
    has_data = ~np.array([np.any(np.isnan(df)) for df in drug_features])
    drug_features = np.stack(np.array(drug_features, dtype=object)[has_data]).astype(
        float
    )
    drugs = drugs[has_data]

    y_train = drug_response_matrices["train"].melt(ignore_index=False).dropna()
    y_test = drug_response_matrices["test"].melt(ignore_index=False).dropna()
    y_val = drug_response_matrices["validation"].melt(ignore_index=False).dropna()

    y_train = y_train.loc[y_train.index.isin(cell_lines) & y_train["drugs"].isin(drugs)]
    y_test = y_test.loc[y_test.index.isin(cell_lines) & y_test["drugs"].isin(drugs)]
    y_val = y_val.loc[y_val.index.isin(cell_lines) & y_val["drugs"].isin(drugs)]

    y_train = y_train.reset_index().rename(columns={"value": "response"})
    y_test = y_test.reset_index().rename(columns={"value": "response"})
    y_val = y_val.reset_index().rename(columns={"value": "response"})

    def get_cell_line_features(y):
        cell_line_features = [
            cell_line_name_to_gene_expression_map[cl] for cl in y["cell_lines"]
        ]
        return np.stack(cell_line_features).astype(float)

    def get_drug_features(y):
        drug_features = [drug_name_to_fingerprint_map[d] for d in y["drugs"]]
        return np.stack(drug_features).astype(float)

    train_cell_line_features = get_cell_line_features(y_train)
    test_cell_line_features = get_cell_line_features(y_test)
    val_cell_line_features = get_cell_line_features(y_val)

    # standardize cell line features
    scaler = StandardScaler()
    scaler.fit(train_cell_line_features)
    train_cell_line_features = scaler.transform(train_cell_line_features)
    test_cell_line_features = scaler.transform(test_cell_line_features)
    val_cell_line_features = scaler.transform(val_cell_line_features)

    train_drug_features = get_drug_features(y_train)
    test_drug_features = get_drug_features(y_test)
    val_drug_features = get_drug_features(y_val)

    # join cell line and drug features
    train_features = np.concatenate(
        [train_cell_line_features, train_drug_features], axis=1
    )
    test_features = np.concatenate(
        [test_cell_line_features, test_drug_features], axis=1
    )
    val_features = np.concatenate([val_cell_line_features, val_drug_features], axis=1)

    return {
        "train": {"features": train_features, "targets": y_train, "scaler": scaler},
        "test": {"features": test_features, "targets": y_test},
        "validation": {"features": val_features, "targets": y_val},
        "gene_list": gene_list,
    }

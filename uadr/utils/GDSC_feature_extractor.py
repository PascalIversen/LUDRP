import pandas as pd
import io, platform

from pandas._libs.parsers import ParserError


def get_gdsc_data(
    response_type="LN_IC50",
    path_gdsc1="data/GDSC1_fitted_dose_response_15Oct19.xlsx",
    path_gdsc2="data/GDSC2_fitted_dose_response_15Oct19.xlsx",
    path_cell_annotations="data/Cell_Lines_Details.xlsx",
    path_drug_annotations="data/screened_compounds_rel_8.1.csv",
):
    """
    get a DataFrame with the response and some cell-line/drug features of the GDSC1 and GDSC2 databases

    response_types: LN_IC50 (natural logarithm of the IC50
    Z_SCORE (drug-wise Z-scaled LN_IC50)
    AUC (area under the dose-response curve)
    path_gdsc1: path of GDSC1-dataset
    path_gdsc2: path of GDSC1-dataset
    path_cell_annotations: path of GDSC cell-line annotations
    path_drug_annotations: path of GDSC drug annotations
    get from:
    https://www.cancerrxgene.org/downloads/bulk_download

    """

    data_gdsc1 = pd.read_excel(path_gdsc1)
    data_gdsc2 = pd.read_excel(path_gdsc2)

    # create an unique id column
    data_gdsc1["experiment_id"] = (
        data_gdsc1["DRUG_NAME"] + "_" + data_gdsc1["CELL_LINE_NAME"]
    )
    data_gdsc2["experiment_id"] = (
        data_gdsc2["DRUG_NAME"] + "_" + data_gdsc2["CELL_LINE_NAME"]
    )

    data_gdsc1 = data_gdsc1.set_index("experiment_id")
    data_gdsc2 = data_gdsc2.set_index("experiment_id")

    # combine GDSC1 and GDSC2, if an experiment was redone for GDSC2, use the new data
    data_gdsc = data_gdsc2.combine_first(data_gdsc1)

    # keep only the response specified
    response_types = ["LN_IC50", "Z_SCORE", "AUC"]
    try:
        response_types.remove(response_type)
    except:
        print("Invalid response type. Use one of LN_IC50, Z_SCORE, AUC")

    data_gdsc = data_gdsc.drop(response_types, axis=1)

    # get GDSC cell-line features

    cell_line_data = pd.read_excel(path_cell_annotations)

    unique_features = list(
        cell_line_data.columns.difference(data_gdsc.columns)
    )  # only add new features
    data_gdsc = pd.merge(
        left=data_gdsc,
        right=cell_line_data[unique_features],
        right_on="Sample Name",
        left_on="CELL_LINE_NAME",
        how="left",
    )

    # get GDSC drug properties
    drug_data = pd.read_csv(path_drug_annotations)
    unique_features = list(
        drug_data.columns.difference(data_gdsc.columns)
    )  # only add new features
    unique_features.append("DRUG_NAME")

    data_gdsc = pd.merge(
        left=data_gdsc,
        right=drug_data[unique_features],
        right_on="DRUG_NAME",
        left_on="DRUG_NAME",
        how="left",
    )

    # delete columns that are not useful
    useless_columns = [
        "DATASET",
        "NLME_RESULT_ID",
        "NLME_CURVE_ID",
        "COSMIC_ID",
        "SANGER_MODEL_ID",
        "TCGA_DESC",
        "DRUG_ID",
        "COMPANY_ID",
        "WEBRELEASE",
        "MIN_CONC",
        "MAX_CONC",
        "RMSE",
        "Sample Name",
        "COSMIC identifier",
    ]
    data_gdsc = data_gdsc.drop(columns=useless_columns)

    return data_gdsc


# In[ ]:


# In[1]:


def get_gdsc_mutations(
    cell_line_names, kernel="jaccard", path_mutations="data/mutations_20191101.csv"
):
    """
    get gene mutation similiarity matrix of cell_lines or get the gene mutation feature matrix
    (if kernel=None)

    returns:
    pandas DataFrame: cell_line_names x cell_line_names (or if kernel= None: cell_line_names x genes)

    if there is no copy number data for a cell_line in cell_line_names, the cell is not part of the returned DataFrame

    cell_line_names:
    list of str: GDSC cell-line names


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

    cells_not_in_input = set(mutations.index) - set(cell_line_names)
    cells_not_in_data = set(cell_line_names) - set(mutations.index)

    # remove cells for which we found no data
    cell_line_names = set(cell_line_names) & set(mutations.index)

    print("For " + str(len(cells_not_in_data)) + " cells, no mutation data found...")

    # drop rows: cells for which were not queried
    mutations = mutations.drop(cells_not_in_input, axis=0)

    if kernel == None:
        return mutations

    elif kernel == "jaccard":
        similarity = pd.DataFrame(columns=cell_line_names, index=cell_line_names)

        from scipy.spatial.distance import jaccard
        from tqdm._tqdm_notebook import tqdm_notebook

        for cell_from in tqdm_notebook(cell_line_names):
            for cell_to in cell_line_names:
                # make sure to not have integers
                cell_from = str(cell_from)
                cell_to = str(cell_to)

                try:
                    similarity[cell_from].loc[cell_to] = 1 - jaccard(
                        mutations.loc[cell_from], mutations.loc[cell_to]
                    )
                    similarity[cell_to].loc[cell_from] = similarity[cell_from].loc[
                        cell_to
                    ]
                except KeyError:
                    pass
        return similarity

    else:
        raise ValueError(""" kernel has to be "jaccard" or None """)


# In[220]:


# TODO ask for difference of picnic and gistic


def get_gdsc_copy_number_var(
    cell_line_names, data_type="gistic", kernel="rbf", rbf_gamma=None
):
    """
    get copy number variation similiarity matrix of cell_lines or get the cell_line copy number variation feature matrix
    (if kernel=None)

    returns:
    pandas DataFrame: cell_line_names x cell_line_names (or if kernel= None: cell_line_names x genes)

    if there is no copy number data for a cell_line in cell_line_names, the cell is not part of the returned DataFrame

    cell_line_names:
    list of str: GDSC cell-line names

    data_type:
    use absolute PICNIC or GISTIC data, get from: https://cellmodelpassports.sanger.ac.uk/downloads

    kernel:
    "pearson_corr": Pearson correlation coefficients of the gene expression profiles
    "rbf": Radial basis function similiarity of the gene expression profiles
    None: return copy number variation feature matrix

    rbf_gamma :
    float, gamma parameter of the RBF, if None, defaults to 1.0 / n_features
    """

    if data_type == "picnic":
        path_cnv = "data/GDSC/cnv_abs_copy_number_picnic_20191101.csv"
    elif data_type == "gistic":
        path_cnv = "data/GDSC/cnv_gistic_20191101.csv"
    else:
        raise ValueError("""data_type has to be "picnic" or "gistic" """)

    # make feature matrix:

    sanger_ids = pd.Series(
        cell_line_names_to_sanger_ids(cell_line_names), index=cell_line_names
    )

    print(
        "SangerId of " + str(sanger_ids.isna().sum()) + " cell line names are unknown."
    )
    sanger_ids = sanger_ids.dropna()

    cnv_norm = pd.read_csv(path_cnv, header=[0, 1], index_col=[0, 1])

    cnv_norm.columns = cnv_norm.columns.droplevel(-1)

    cnv_feature = pd.DataFrame()

    n_not_found = 0
    for sanger_id, cell in zip(sanger_ids, cell_line_names):
        try:
            cnv_feature[cell] = cnv_norm[sanger_id]
        except KeyError:
            n_not_found += 1

    print("For " + str(n_not_found) + " cells, no CNV data was found")

    cnv_feature = cnv_feature.transpose()

    if kernel == None:
        return cnv_feature

    elif kernel == "pearson_corr":
        similiarity_matrix = cnv_feature.T.corr()
        return similiarity_matrix

    elif kernel == "rbf":
        from sklearn.metrics.pairwise import rbf_kernel

        # if any value is nan fill with mean
        if cnv_feature.isna().sum().sum() > 0:
            # TODO mabye think of using generalized RBF instead
            print(
                "Warning: nan values in the CNV data, impute with the mean, watch out for target leakage"
            )
            cnv_feature.fillna(cnv_feature.mean(), inplace=True)
        similiarity_matrix = rbf_kernel(X=cnv_feature, gamma=rbf_gamma)
        similiarity_matrix = pd.DataFrame(
            similiarity_matrix, index=cnv_feature.index, columns=cnv_feature.index
        )
        return similiarity_matrix

    else:
        raise ValueError(
            """ Invalid kernel specified, choose "rbf" or "pearson_corr" or None (not  "None") """
        )


# In[7]:


def cell_line_names_to_sanger_ids(
    cell_line_names, update=False, path_to_dict="data/cell_line_name_to_sanger_id.npy"
):
    """
    map cell_line_names to sanger_ids using GDSC data
    returns: list of sanger ids for the cell line names, containing np.nan for unkown cell-line-names

    if update=True the dict in the data folder gets updated using the GDSC data in the data folder

    """

    import numpy as np

    # Load
    cell_line_name_to_sanger_id_map = np.load(path_to_dict, allow_pickle="TRUE").item()

    # update to dictionary for new cell lines using the gdsc1 and gdsc2 data
    if update:
        path_gdsc1 = "data/GDSC1_fitted_dose_response_15Oct19.xlsx"
        path_gdsc2 = "data/GDSC2_fitted_dose_response_15Oct19.xlsx"
        data_gdsc1 = pd.read_excel(path_gdsc1)
        data_gdsc2 = pd.read_excel(path_gdsc2)

        # create an unique id column
        data_gdsc1["experiment_id"] = (
            data_gdsc1["DRUG_NAME"] + "_" + data_gdsc1["CELL_LINE_NAME"]
        )
        data_gdsc2["experiment_id"] = (
            data_gdsc2["DRUG_NAME"] + "_" + data_gdsc2["CELL_LINE_NAME"]
        )

        data_gdsc1 = data_gdsc1.set_index("experiment_id")
        data_gdsc2 = data_gdsc2.set_index("experiment_id")

        # combine GDSC1 and GDSC2, if an experiment was redone for GDSC2, use the new data
        data_gdsc = data_gdsc2.combine_first(data_gdsc1)

        # update the dictionary
        seen = list(cell_line_name_to_sanger_id_map.keys())

        for cell, sanger_id in zip(
            list(data_gdsc["CELL_LINE_NAME"]), list(data_gdsc["SANGER_MODEL_ID"])
        ):
            if not (cell in seen):
                seen.append(cell)
                cell_line_name_to_sanger_id_map[cell] = sanger_id

        np.save(path_to_dict, cell_line_name_to_sanger_id_map)

    # use the dictionary to map cell line names to sanger id, append np.nan if not found in dict
    sanger_ids = []
    for cell in cell_line_names:
        try:
            sanger_ids.append(cell_line_name_to_sanger_id_map[cell])
        except KeyError:
            sanger_ids.append(np.nan)

    return sanger_ids


# In[ ]:


# In[36]:


def get_inchi_keys(
    compound_names=[],
    synonyms_list=[],
    path_to_inchi_key_data="data/compounds_inchi_key.csv",
):
    """
    get inchi_keys from existing data and request missing InCHIs
    from PubChem and transform to InChi-key by requesting Chemspider

    returns: series of inchi_key strs indexed by drug names

    compound names are a list of str
    synonyms (optional) is a list of str synonyms for the drug names seperated by ","
    e.g ["d1synonym1,d1synonym2", "d2synonym1", "", ...]

    path_to_inchi_key_data: path of csv containing columns 'inchi_key' and 'drug_name'
    """

    import pubchempy as pcp
    import requests
    import numpy as np
    import tqdm

    # if no synonyms are given:
    if len(synonyms_list) == 0:
        synonyms_list = [[]] * len(compound_names)
    elif len(compound_names) != len(synonyms_list):
        print(
            "compound_names and synonyms_list must have same length if synonyms_list is not empty []"
        )
        assert len(compound_names) == len(synonyms_list)

    # because we have compounds with more than one synonym:
    synonyms_list = pd.Series(synonyms_list).str.split(",")
    # empty list for compounds without synonyms:
    synonyms_list.loc[synonyms_list.isnull()] = synonyms_list.loc[
        synonyms_list.isnull()
    ].apply(lambda x: [])

    inchi_key_result = pd.DataFrame(
        {"synonyms": list(synonyms_list)}, index=compound_names
    )

    # remove duplicates from the input compound_names:
    if inchi_key_result.index.name == None:
        index_name = "index"
    else:
        index_name = inchi_key_result.index.name
    inchi_key_result = (
        inchi_key_result.reset_index()
        .drop_duplicates(subset=index_name)
        .set_index(index_name)
    )

    # get existing inchi_key drug data
    if path_to_inchi_key_data != None:
        inchi_key_data = pd.read_csv(path_to_inchi_key_data)
        inchi_key_data = inchi_key_data.set_index("drug_name")
        inchi_key_result = inchi_key_result.merge(
            inchi_key_data["inchi_key"], left_index=True, right_index=True, how="left"
        )

    missing_drugs = inchi_key_result[inchi_key_result["inchi_key"].isna()].loc[
        :, "synonyms"
    ]

    for compound_name, synonyms in zip(list(missing_drugs.index), list(missing_drugs)):
        # search compound on PubChem
        compounds = pcp.get_compounds(compound_name, "name")

        if len(compounds) > 1:
            print(
                "Compound: "
                + compound_name
                + " Warning: "
                + str(len(compounds))
                + " results found for query on PubChem"
            )
            print("taking first compound found")
            inchi = compounds[0].to_dict(properties=["inchi"])["inchi"]

        elif len(compounds) == 0:
            if len(synonyms) == 0:
                print(
                    "Compound: "
                    + compound_name
                    + " Warning: no results found on PubChem and no synonyms given"
                )
                continue  # end the loop for this compound

            else:
                has_found_compound = False
                for synonym in synonyms:
                    compounds = pcp.get_compounds(synonym, "name")
                    if len(compounds) > 0:
                        inchi = compounds[0].to_dict(properties=["inchi"])["inchi"]
                        print(
                            "Compound: "
                            + compound_name
                            + " found as synonym "
                            + synonym
                        )
                        has_found_compound = True
                        break  # break the search via synonyms, since we found the compound
                if not has_found_compound:
                    print(
                        "Compound: "
                        + compound_name
                        + " Warning: no results found on PubChem"
                    )
                    continue  # end the loop for this compound

        else:
            inchi = compounds[0].to_dict(properties=["inchi"])["inchi"]

        # retrieve InChiKey from chemspider
        host = "http://www.chemspider.com"
        getstring = "/InChI.asmx/InChIToInChIKey?inchi="

        r = requests.get("{}{}{}".format(host, getstring, inchi))
        if r.ok:
            inchikey = str(
                r.text.replace(
                    '<?xml version="1.0" encoding="utf-8"?>\r\n<string xmlns="http://www.chemspider.com/">',
                    "",
                )
                .replace("</string>", "")
                .strip()
            )
            inchi_key_result.loc[compound_name, "inchi_key"] = inchikey
        else:
            print("invalid inchi for compound: " + str(compound_name))

    print(
        "Found: "
        + str(
            len(inchi_key_result["inchi_key"])
            - inchi_key_result["inchi_key"].isna().sum()
        )
        + " of "
        + str(len(inchi_key_result["inchi_key"]))
        + " inchi keys."
    )

    # update inchi key csv file
    if path_to_inchi_key_data != None:
        df_data = inchi_key_data[~inchi_key_data["inchi_key"].isna()]
        df_result = inchi_key_result[~inchi_key_result["inchi_key"].isna()]
        compounds_inchi_key = pd.concat(
            [df_data["inchi_key"], df_result["inchi_key"]], sort=False
        )

        compounds_inchi_key.index.name = "drug_name"

        compounds_inchi_key = (
            compounds_inchi_key.reset_index()
            .drop_duplicates(subset="drug_name")
            .set_index("drug_name")
        )
        compounds_inchi_key.to_csv(path_to_inchi_key_data)

    return pd.Series(inchi_key_result.drop("synonyms", axis=1)["inchi_key"])


# In[221]:


def cosmic_ids_to_cell_line_names(
    cosmic_ids, path_cell_annotations="data/GDSC/Cell_Lines_Details.csv", verbose=False
):
    """
    transform a list of COSMIC ID's to a series of cell-line-names, indexed by the cosmic ID
    using the cell annotations from https://www.cancerrxgene.org/downloads/bulk_download

    """

    try:
        cell_line_data = pd.read_csv(path_cell_annotations, index_col=0)
    except ParserError:
        csv_data = open(path_cell_annotations).read().replace("\r\n", "\n")
        cell_line_data = pd.read_csv(io.StringIO(csv_data), encoding="unicode_escape")

    cosmic_ids_to_cell_line_name_dict = pd.Series(
        cell_line_data["Sample Name"].values,
        index=cell_line_data["COSMIC identifier"].fillna(-1).astype(int).values,
    ).to_dict()

    cell_line_names = []
    unknown_cell_line_names = []
    for cosmic_id in cosmic_ids:
        try:
            cell_line_names.append(cosmic_ids_to_cell_line_name_dict[int(cosmic_id)])

        except (KeyError, ValueError):
            cell_line_names.append("unknown_cosmic_" + str(cosmic_id))
            unknown_cell_line_names.append(cosmic_id)

    if unknown_cell_line_names and verbose:
        print(
            "Note: "
            + str(len(unknown_cell_line_names))
            + " Cosmic IDs not found in cell annotation data: "
        )
        print(unknown_cell_line_names)

    # check if cell_line_names are unique
    unique_c = []
    dup_c = []
    for c in cell_line_names:
        if not (c in unique_c):
            unique_c.append(c)
        else:
            dup_c.append(c)
    if dup_c:
        print(
            "Warning: at least two cosmic IDs map to the same cell lines for the cell lines: "
        )
        print(dup_c)

    return pd.Series(cell_line_names, index=cosmic_ids)


def get_gdsc_gene_expression(
    cell_line_names,
    path_gene_expression="data/Cell_line_RMA_proc_basalExp.txt",
    path_cell_annotations="data/Cell_Lines_Details.xlsx",
    kernel="pearson_corr",
    rbf_gamma=None,
    scaling="minmax",
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
    import pandas as pd

    # gene expression data
    try:
        gene_expression = pd.read_csv(path_gene_expression, sep=";", decimal=",")
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

    # only take cell-lines for which we have data and only take gene expression data, if it was queried:
    columns_to_keep = set(cell_line_names) & set(ge_columns)
    columns_to_drop = set(ge_columns) - columns_to_keep
    gene_expression = gene_expression.drop(columns_to_drop, axis=1)

    print()
    print(
        "Gene expression data available for "
        + str(len(columns_to_keep))
        + " of the "
        + str(len(cell_line_names))
        + " queried cell-lines."
    )

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


# In[40]:


def get_padel_drug_features(
    inchi_keys,
    get_descriptors=True,
    get_fingerprints=True,
    chemspider_api_key="kb5NIxq1biG0aka4AF86HGKpO5ZEp508",
    timeout=200,
):
    """
    get PaDEL descriptors and fingerprints for each compound by inchi_keys

    returns:
    pandas DataFrame.
    columns:= descriptor + fingerprints
    rows := compound inchi_keys

    inchi_key: pandas Series of inchi_keys-str of drugs indexed by drug name for which to get PaDEL data
    get_descriptors: bool, if False, do not get descriptors
    get_fingerprints: bool, if False, do not get fingerprints
    chemspider_api_key: string, if default key is outdated get from https://developer.rsc.org/


    """

    import pandas as pd
    import numpy as np

    assert isinstance(inchi_keys, pd.Series)

    inchi_keys.index.name = "index"

    try:
        from padelpy import from_smiles
    except ImportError:
        print("please install padelpy: !pip install padelpy")
        raise
    try:
        import chemspipy
    except ImportError:
        print("please install chemspipy: !pip install chemspipy")
        raise

    # remove duplicates:
    inchi_keys = inchi_keys.drop_duplicates()

    # convert inchi keys to SMILES using identifier data from a file
    # if not present use chemspider API to query SMILES

    # get smiles representation from csv
    ex_compound_identifier = pd.read_csv("data/compound_identifier.csv").set_index(
        "InChIKey"
    )

    # get smiles codes via Chemspider api
    cs = chemspipy.ChemSpider(chemspider_api_key)

    smiles = pd.Series()
    for inchi_key, drug_name in zip(list(inchi_keys), list(inchi_keys.index)):
        try:
            smiles[drug_name] = ex_compound_identifier.loc[inchi_key, "SMILES"]
        except KeyError:
            try:
                inchi = cs.convert(inchi_key, "InChIKey", "InChI")
                smiles_repr = cs.convert(inchi, "InChI", "SMILES")

                # update data
                ex_compound_identifier.loc[inchi_key, "InChI"] = inchi
                ex_compound_identifier.loc[inchi_key, "SMILES"] = smiles_repr
                ex_compound_identifier.loc[inchi_key, "drug_name"] = drug_name

                smiles[drug_name] = smiles_repr

            except chemspipy.errors.ChemSpiPyBadRequestError:
                print(
                    "Could not convert Inchi-Key "
                    + str(inchi_key)
                    + " to SMILES. This compound will be missing from the result."
                )
            except chemspipy.errors.ChemSpiPyRateError:
                print(
                    "Too many requests to chemspider, get new API Key from https://developer.rsc.org/"
                )
                raise

    # drugs, for which we found SMILES:
    valid_drugs = list(smiles.index)

    # update CSV file
    ex_compound_identifier = ex_compound_identifier.reset_index()
    ex_compound_identifier.to_csv("data/compound_identifier.csv")
    del ex_compound_identifier

    # get fingerprints and/or descriptors for the smiles codes
    drug_features = pd.DataFrame(columns=valid_drugs)

    from tqdm._tqdm_notebook import tqdm_notebook

    print("fetching features...")
    for drug in tqdm_notebook(valid_drugs):
        try:
            fts = from_smiles(
                smiles.loc[drug],
                fingerprints=get_fingerprints,
                descriptors=get_descriptors,
                timeout=timeout,
            )
        except RuntimeError:
            print("""Runtime Error: Increase timeout value (default: timeout=200) """)
            raise

        drug_features[drug] = pd.Series(fts)

    # convert from string to numeric
    drug_features = drug_features.replace("Infinity", np.nan)
    drug_features = drug_features.apply(lambda y: pd.to_numeric(y, errors="coerce"))

    return drug_features.transpose()


# In[42]:


def drug_similiarity(
    drug_features, kernel="pearson_corr", scaling="minmax", rbf_gamma=None
):
    """
    calculate the similiarity matrix between drugs

    returns:
    pandas DataFrame drugs x drugs

    drug_inputs: pandas DataFrame columns: features, rows: drugs
    kernel: similarity measure, can be "pearson_corr", "rbf"
    scaling: normalizer of the features, can be "minmax" or "None" for no scaling
    rbf_gamma: float, gamma parameter of the RBF, if None, defaults to 1.0 / n_features

    """
    if scaling == "minmax":
        from sklearn.preprocessing import minmax_scale

        drug_features = pd.DataFrame(
            minmax_scale(drug_features),
            columns=drug_features.columns,
            index=drug_features.index,
        )

    if kernel == "pearson_corr":
        import numpy as np

        similiarity_matrix = drug_features.T.corr()

    elif kernel == "rbf":
        from sklearn.metrics.pairwise import rbf_kernel

        # if any value is nan fill with mean
        if drug_features.isna().sum().sum() > 0:
            print(
                "Warning: nan values in the drug feature data, impute with the mean, watch out for target leakage"
            )
            drug_features.fillna(drug_features.mean(), inplace=True)

        similiarity_matrix = rbf_kernel(X=drug_features, gamma=rbf_gamma)
        similiarity_matrix = pd.DataFrame(
            similiarity_matrix, index=drug_features.index, columns=drug_features.index
        )
    else:
        raise ValueError(
            """ Invalid kernel specified, choose "rbf" or "pearson_corr" """
        )

    return similiarity_matrix


# In[356]:


def get_gdsc_drug_target_matrix(
    path_drug_annotations="data/screened_compounds_rel_8.1.csv",
):
    """
    get binary drug-target matrix for the GDSC drugs

    Returns: pandas Dataframe: drugs x targets

    path_drug_annotations: path of GDSC drug annotations
    get data from:
    https://www.cancerrxgene.org/downloads/bulk_download
    """

    from functools import reduce

    drug_data = pd.read_csv(path_drug_annotations)
    drug_data = drug_data.set_index("DRUG_NAME")

    drug_targets = (
        drug_data["PUTATIVE_TARGET"]
        .fillna("nan")
        .apply(lambda x: x.replace(" ", "").split(","))
    )

    def combine(x, y):
        x.extend(y)
        return x

    drug_targets = drug_targets.loc[~drug_targets.index.duplicated()]
    targets = set(reduce(combine, drug_targets))
    drugs = set(drug_targets.index)

    drug_target_matrix = pd.DataFrame(columns=targets, index=drugs)

    for drug in drugs:
        # for every target of a drug create entry in the matrix
        for target in drug_targets.loc[drug]:
            drug_target_matrix.loc[drug, target] = 1

    drug_target_matrix = drug_target_matrix.fillna(0)

    return drug_target_matrix

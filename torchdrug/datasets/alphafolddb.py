import os
import glob

from torchdrug import data, utils
from torchdrug.core import Registry as R


@R.register("datasets.AlphaFoldDB")
@utils.copy_args(data.ProteinDataset.load_pdbs)
class AlphaFoldDB(data.ProteinDataset):
    """
    3D protein structures predicted by AlphaFold.
    This dataset covers proteomes of 48 organisms, as well as the majority of Swiss-Prot.

    Statistics:
        See https://alphafold.ebi.ac.uk/download

    Parameters:
        path (str): path to store the dataset
        species_id (int, optional): the id of species to be loaded. The species are numbered
            by the order appeared on https://alphafold.ebi.ac.uk/download (0-20 for model 
            organism proteomes, 21 for Swiss-Prot)
        split_id (int, optional): the id of split to be loaded. To avoid large memory consumption 
            for one dataset, we have cut each species into several splits, each of which contains 
            at most 22000 proteins.
        verbose (int, optional): output verbose level
        **kwargs
    """

    urls = [
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000006548_3702_ARATH_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000001940_6239_CAEEL_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000000559_237561_CANAL_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000000437_7955_DANRE_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000002195_44689_DICDI_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000000803_7227_DROME_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000000625_83333_ECOLI_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000008827_3847_SOYBN_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000005640_9606_HUMAN_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000008153_5671_LEIIN_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000000805_243232_METJA_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000000589_10090_MOUSE_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000001584_83332_MYCTU_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000059680_39947_ORYSJ_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000001450_36329_PLAF7_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000002494_10116_RAT_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000002311_559292_YEAST_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000002485_284812_SCHPO_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000008816_93061_STAA8_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000002296_353153_TRYCC_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000007305_4577_MAIZE_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/swissprot_pdb_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000001631_447093_AJECG_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000006672_6279_BRUMA_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000000799_192222_CAMJE_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000094526_86049_9EURO1_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000274756_318479_DRAME_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000325664_1352_ENTFC_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000053029_1442368_9EURO2_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000000579_71421_HAEIN_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000000429_85962_HELPY_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000007841_1125630_KLEPH_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000008153_5671_LEIIN_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000078237_100816_9PEZI1_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000000806_272631_MYCLE_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000001584_83332_MYCTU_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000020681_1299332_MYCUL_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000000535_242231_NEIG1_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000006304_1133849_9NOCA1_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000024404_6282_ONCVO_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000002059_502779_PARBA_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000001450_36329_PLAF7_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000002438_208964_PSEAE_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000001014_99287_SALTY_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000008854_6183_SCHMA_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000002716_300267_SHIDS_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000018087_1391915_SPOS1_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000008816_93061_STAA8_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000000586_171101_STRR6_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000035681_6248_STRER_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000030665_36087_TRITR_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000008524_185431_TRYB2_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000002296_353153_TRYCC_v2.tar",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000270924_6293_WUCBA_v2.tar"
    ]
    md5s = [
        "4cd5f596ebfc3d45d9f6b647dc5684af", "9e26602ba2d9f233ef4fcf82703ddb59",
        "60a09db1e1c47a98763d09879784f536", "a0ab562b7372f149673c4518f949501f", 
        "6205138b14fb7e7ec09b366e3e4f294b", "31f31359cd7254f82304e3886440bdd3", 
        "a590096e65461ed4eb092b2147b97f0b", "8f1e120f372995644a7101ad58e5b2ae", 
        "9a659c4aed2a8b833478dcd5fffc5fd8", "95d775f2ae271cf50a101c73335cd250", 
        "e5b12da43f5bd77298ca50e19706bdeb", "90e953abba9c8fe202e0adf825c0dfcc", 
        "38a11553c7e2d00482281e74f7daf321", "2bcdfe2c37154a355fe4e8150c279c13", 
        "580a55e56a44fed935f0101c37a8c4ab", "b8d08a9033d111429fadb4e25820f9f7", 
        "59d1167f414a86cbccfb204791fea0eb", "dfde6b44026f19a88f1abc8ac2798ce6", 
        "a1c2047a16130d61cac4db23b2f5b560", "e4d4b72df8d075aeb607dcb095210304", 
        "5cdad48c799ffd723636cae26433f1f9", "98a7c13987f578277bfb66ac48a1e242", 
    ]
    species_nsplit = [
        2, 1, 1, 2, 1, 1, 1, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 20,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    ]
    split_length = 22000

    def __init__(self, path, species_id=0, split_id=0, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        species_name = os.path.basename(self.urls[species_id])[:-4]
        if split_id >= self.species_nsplit[species_id]:
            raise ValueError("Split id %d should be less than %d in species %s" % 
                            (split_id, self.species_nsplit[species_id], species_name))
        self.processed_file = "%s_%d.pkl.gz" % (species_name, split_id)
        pkl_file = os.path.join(path, self.processed_file)

        if os.path.exists(pkl_file):
            self.load_pickle(pkl_file, verbose=verbose, **kwargs)
        else:
            tar_file = utils.download(self.urls[species_id], path, md5=self.md5s[species_id])
            pdb_path = utils.extract(tar_file)
            gz_files = sorted(glob.glob(os.path.join(pdb_path, "*.pdb.gz")))
            pdb_files = []
            index = slice(split_id * self.split_length, (split_id + 1) * self.split_length)
            for gz_file in gz_files[index]:
                pdb_files.append(utils.extract(gz_file))
            self.load_pdbs(pdb_files, verbose=verbose, **kwargs)
            self.save_pickle(pkl_file, verbose=verbose)

    def get_item(self, index):
        if getattr(self, "lazy", False):
            protein = data.Protein.from_pdb(self.pdb_files[index], self.kwargs)
        else:
            protein = self.data[index].clone()
        if hasattr(protein, "residue_feature"):
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()
        item = {"graph": protein}
        if self.transform:
            item = self.transform(item)
        return item

    def __repr__(self):
        lines = [
            "#sample: %d" % len(self),
        ]
        return "%s(\n  %s\n)" % (self.__class__.__name__, "\n  ".join(lines))

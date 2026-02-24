"""
Multi-label GPCR–G protein coupling dataset.

Loads receptor data from GPCRdb with coupling labels for all four
G protein families (Gs, Gi/o, Gq/11, G12/13).
"""
import os
import json
import csv
import numpy as np
import pandas as pd
from collections import defaultdict

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_DIR, "data")

# Canonical G-alpha sequences for k-mer overlap features
GNAQ_SEQ = (
    "MTLESIMACCLSEEAKEARRINDEIERQLRRDKRDARRELKLLLLGTGESGKSTFIKQMRIIHGS"
    "GYSDEDKRGFTKLVYQNIFTAMQAMIRAMDTLKIPYKYEHNKAHAQLVREVDVEKVSAFENPYVD"
    "AIKSLWNDPGIQECYDRRREYQLSDSTKYYLNDLDRVADPAYLPTQQDVLRVRVPTTGIIEYPFDL"
    "QSVIFRMVDVGGQRSERRKWIHCFENVTSIMFLVALSEYDQVLVESDNENRMEESKALFRTIITYPWFQNSSVILFLNKKDLLEEK"
)
GNAS_SEQ = (
    "MGCLGNSKTEDQRNEEKAQREANKKIEKQLQKDKQVYRATHRLLLLGAGESGKSTIVKQMRILHV"
    "NGFNGEGGEEDPQAARSNSDGEKATKVQDIKNNLKEAIETIVAAMSNLVPPVELANPENQFRVDYIL"
    "SVMNVPDFDFPPEFYEHAKALWEDEGVRACYERSNEYQLIDCAQYFLDKIDVIKQADYVPSDQDLLR"
    "CRVLTSGIFETKFQVDKVNFHMFDVGGQRDERRKWIQCFNDVTAIIFVVASSSYNMVIREDNQTNRL"
    "QEALNLFKSIWNNRWLRTISVILFLNKQDLLAEKVLAGKSKIEDYFPEFARYTTPEDATPEPGEDPRVTRAKYFIRDEFLRISTASGDGRHYCYPHFTCAVDTENIRRVFNDCRDIIQRMHLRQYELL"
)
GNAI_SEQ = (
    "MGCTLSAEDKAAVERSKMIDRNLREDGEKAAREVKLLLLGAGESGKSTIVKQMKIIHEAGYSEEEC"
    "KQYKAVVYSNTIQSIIAIIRAMGRLKIDFGDSARADDARQLFVLAGAAEEGFMTAELAGVIKRLWK"
    "DSGVQACFNRSREYQLNDSAAYYLNDLDRIAQPNYIPTQQDVLRTRVKTTGIVETHFTFKDLHFKMF"
    "DVGGQRSERKKWIHCFEGVTAIIFCVALSDYDLVLAEDEEMNRMHESMKLFDSICNNKWFTDTSIILF"
    "LNKKDLFEEKITHSPLTICFPEYTGANKYDEASYYIQSKFEDLNKRKDTKEIYTHFTCATDTKNVQFVFDAVTDVIIKNNLKDCGLF"
)
GNA12_SEQ = (
    "MSGVVRTLSRCLLPAEAGARERRAGSGARDAEREARRRSRDIDALLARERAVRRLVK"
    "ILLLGAGESGKSTFLKQMRIIHGREFDQKALLEFRDTIFDNILKGSRVLVDARDKLG"
    "IPWQYSENEKHGMFLMAFENKAGLPVEPATFQLYVPALSALWRDSGIREYQLNDSAA"
    "YYLNDLERIAQSDYIPTQQDVLRTRVKTTGIVETHFTFKDLYFKMFDVGGQRSERKK"
    "WIHCFENVITAIIFVVASSSYNMVIREDNQTNRLQEALNLFKSIWNNRWLRTISVILF"
    "LNKQDLLAEKVLAGKSKIEDYFPEFAR"
)

G_PROTEIN_SEQS = {
    "gnaq": GNAQ_SEQ,
    "gnas": GNAS_SEQ,
    "gnai": GNAI_SEQ,
    "gna12": GNA12_SEQ,
}

# All four coupling targets
COUPLING_TARGETS = ["Gs", "Gi", "Gq", "G12"]


class GPCRDataset:
    """Multi-label GPCR–G protein coupling dataset."""

    def __init__(self, data_dir=None):
        self.data_dir = data_dir or DATA_DIR
        self.receptors = []
        self.labels = {}          # target -> np.array
        self.families = []
        self.subfamilies = []
        self.sequences = []
        self.entry_names = []
        self.accessions = []

    def load(self, csv_path=None, json_path=None, exclude_unknown=True):
        """Load dataset from CSV or JSON.

        Returns self for chaining.
        """
        if csv_path is None:
            csv_path = os.path.join(self.data_dir, "gpcrdb_coupling_dataset.csv")
        if json_path is None:
            json_path = os.path.join(self.data_dir, "gpcrdb_coupling_dataset.json")

        # Prefer JSON (richer data), fall back to CSV
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        elif os.path.exists(csv_path):
            raw = pd.read_csv(csv_path).to_dict("records")
        else:
            raise FileNotFoundError(f"No dataset found in {self.data_dir}")

        # Parse multi-label coupling from coupling_description
        for r in raw:
            desc = str(r.get("coupling_description", ""))
            if exclude_unknown and desc in ("Unknown", ""):
                continue
            seq = r.get("sequence", "")
            if not seq or len(seq) < 50:
                continue

            self.entry_names.append(r["entry_name"])
            self.accessions.append(r.get("accession", ""))
            self.sequences.append(seq)

            fam = str(r.get("family", ""))
            self.families.append(fam)
            parts = fam.split("_")
            self.subfamilies.append("_".join(parts[:3]) if len(parts) >= 3 else fam)

            self.receptors.append({
                "entry_name": r["entry_name"],
                "name": r.get("name", ""),
                "accession": r.get("accession", ""),
                "family": fam,
                "coupling_description": desc,
                "seq_length": len(seq),
            })

        # Build multi-label arrays
        n = len(self.receptors)
        gs_labels = np.zeros(n, dtype=int)
        gi_labels = np.zeros(n, dtype=int)
        gq_labels = np.zeros(n, dtype=int)
        g12_labels = np.zeros(n, dtype=int)

        for i, r in enumerate(self.receptors):
            desc = r["coupling_description"]
            if "Gs" in desc:
                gs_labels[i] = 1
            if "Gi" in desc or "Gt" in desc:
                gi_labels[i] = 1
            if "Gq" in desc:
                gq_labels[i] = 1
            if "G12" in desc:
                g12_labels[i] = 1

        self.labels = {
            "Gs": gs_labels,
            "Gi": gi_labels,
            "Gq": gq_labels,
            "G12": g12_labels,
        }

        print(f"Loaded {n} receptors")
        for target, arr in self.labels.items():
            print(f"  {target}: {int(arr.sum())} positive, {n - int(arr.sum())} negative")

        return self

    @property
    def n_receptors(self):
        return len(self.receptors)

    def get_labels(self, target="Gq"):
        """Get binary label array for a specific G protein target."""
        if target not in self.labels:
            raise ValueError(f"Unknown target: {target}. Choose from {list(self.labels.keys())}")
        return self.labels[target]

    def get_multilabel_matrix(self):
        """Get (n_receptors, 4) multi-label matrix [Gs, Gi, Gq, G12]."""
        return np.column_stack([self.labels[t] for t in COUPLING_TARGETS])

    def summary(self):
        """Print dataset summary statistics."""
        n = self.n_receptors
        print(f"\n{'='*60}")
        print(f"GPCR Coupling Dataset Summary")
        print(f"{'='*60}")
        print(f"Total receptors: {n}")
        print(f"Unique subfamilies: {len(set(self.subfamilies))}")

        # Coupling distribution
        ml = self.get_multilabel_matrix()
        n_coupled = ml.sum(axis=1)
        print(f"\nCoupling target distribution:")
        for i, t in enumerate(COUPLING_TARGETS):
            pos = int(ml[:, i].sum())
            print(f"  {t:>4s}: {pos:3d} positive ({100*pos/n:.1f}%)")
        print(f"\nMulti-coupling:")
        for k in range(5):
            cnt = int((n_coupled == k).sum())
            if cnt > 0:
                print(f"  {k} targets: {cnt} receptors")

        # GPCR class distribution
        class_map = defaultdict(int)
        for fam in self.families:
            if fam.startswith("001"):
                class_map["Class A"] += 1
            elif fam.startswith("002"):
                class_map["Class B1"] += 1
            elif fam.startswith("003"):
                class_map["Class B2"] += 1
            elif fam.startswith("004"):
                class_map["Class C"] += 1
            elif fam.startswith("005"):
                class_map["Class F"] += 1
            else:
                class_map["Other"] += 1
        print(f"\nGPCR class distribution:")
        for cls, cnt in sorted(class_map.items()):
            print(f"  {cls}: {cnt}")
        print(f"{'='*60}\n")

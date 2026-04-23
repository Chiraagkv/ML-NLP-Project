import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.feature_loader import load_all_features
from src.models import compare_feature_spaces, summary_table
import json
import argparse



def make_json_safe(results):
    cleaned = []
    for r in results:
        r_copy = r.copy()

        # remove non-serializable fields
        r_copy.pop("fitted_model", None)

        # convert numpy arrays → lists
        if "confusion_matrix" in r_copy:
            r_copy["confusion_matrix"] = r_copy["confusion_matrix"].tolist()

        cleaned.append(r_copy)

    return cleaned

parser = argparse.ArgumentParser()
parser.add_argument("--feature_dir", default="./features")
args = parser.parse_args()

experiments, y_train, y_test = load_all_features(args.feature_dir)

all_results = compare_feature_spaces(experiments, y_train, y_test)
summary_table(all_results)


safe_results = make_json_safe(all_results)
with open("./all_results.json", "w") as f:
    json.dump(safe_results, f, indent=2)

print("Saved all_results.json")